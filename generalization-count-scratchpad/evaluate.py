"""Evaluate one checkpoint on seen-position validation and unseen-position test.

Without a scratchpad, the model reads the whole input and predicts the count in one
forward pass.  With a scratchpad (dummy or meaningful), evaluation is free-running:
the model writes its own state after every body token and keeps it in the context,
then answers after ':'.  The primary metric is held-out accuracy on counts 1-3: an
input with no X has no X position, so count 0 is the same in both pools and is
reported separately.
"""
import csv
import os
import sys
from ast import literal_eval
from datetime import datetime

import torch
from torch.nn import functional as F

from dataset import load_meta, load_raw_examples
from engine import MicroTransformer, MicroTransformerConfig
from scratchpad import (
    allowed_state_tokens,
    check_condition,
    gold_state_tokens,
    initial_state_token,
)


# ---------------------------- config (overridable) ----------------------------
out_dir = 'out'
checkpoint_name = 'ckpt.pt'
data_dir = 'data/count_scratchpad'
device = 'cpu'
inference_batch_size = 1024
results_csv = 'results.csv'
predictions_csv = 'predictions.csv'
log_results = True
log_predictions = True
# ------------------------------------------------------------------------------


for arg in sys.argv[1:]:
    if '=' not in arg:
        if arg.startswith('--'):
            raise ValueError(f"expected a config file, got {arg!r}")
        exec(open(arg).read())
    else:
        if not arg.startswith('--'):
            raise ValueError(f"expected --key=value, got {arg!r}")
        key, value = arg[2:].split('=', 1)
        if key not in globals():
            continue
        try:
            value = literal_eval(value)
        except (SyntaxError, ValueError):
            pass
        globals()[key] = value

meta = load_meta(data_dir)
stoi, itos = meta['stoi'], meta['itos']
count_tokens = [str(count) for count in meta['counts']]

checkpoint = torch.load(
    os.path.join(out_dir, checkpoint_name), map_location=device, weights_only=False
)
condition = checkpoint['condition']
check_condition(condition)
train_seed = checkpoint['seed']
checkpoint_iteration = checkpoint['iteration']
pos_type = checkpoint['model_args']['pos_type']
for key in ('length', 'seen_end', 'data_seed'):
    if checkpoint['data_meta'][key] != meta[key]:
        raise ValueError(
            f"checkpoint was trained with {key}={checkpoint['data_meta'][key]}, "
            f"but {data_dir} has {key}={meta[key]}"
        )

model = MicroTransformer(MicroTransformerConfig(**checkpoint['model_args']))
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()


@torch.no_grad()
def predict(examples):
    """Greedy answers, written state traces, and answer cross-entropy."""
    predictions = []
    traces = []
    loss_sum = 0.0
    for start in range(0, len(examples), inference_batch_size):
        batch = examples[start:start + inference_batch_size]
        if condition == 'no_scratchpad':
            contexts = torch.tensor(
                [[stoi[token] for token in body + ':'] for body, _ in batch],
                dtype=torch.long,
                device=device,
            )
            traces.extend('' for _ in batch)
        else:
            contexts = torch.full(
                (len(batch), 1), stoi[initial_state_token(condition)],
                dtype=torch.long, device=device,
            )
            state_columns = []
            for index in range(meta['length']):
                raw_ids = torch.tensor(
                    [stoi[body[index]] for body, _ in batch],
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(1)
                contexts = torch.cat((contexts, raw_ids), dim=1)
                state_ids = model(contexts)[:, -1, :].argmax(-1)
                state_columns.append(state_ids)
                contexts = torch.cat((contexts, state_ids.unsqueeze(1)), dim=1)
            colon = torch.full((len(batch), 1), stoi[':'], dtype=torch.long, device=device)
            contexts = torch.cat((contexts, colon), dim=1)
            state_matrix = torch.stack(state_columns, dim=1).tolist()
            traces.extend(
                ''.join(itos[int(token_id)] for token_id in row) for row in state_matrix
            )
        gold = torch.tensor([stoi[label] for _, label in batch], device=device)
        logits = model(contexts)[:, -1, :]
        loss_sum += F.cross_entropy(logits, gold, reduction='sum').item()
        predictions.extend(itos[int(token_id)] for token_id in logits.argmax(-1).tolist())
    return predictions, traces, loss_sum / len(examples)


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else None


def split_metrics(examples, predictions, traces, answer_loss):
    columns = count_tokens + ['other']
    confusion = {gold: {pred: 0 for pred in columns} for gold in count_tokens}
    per_count = {gold: [0, 0] for gold in count_tokens}
    under = over = invalid = 0

    for (_, gold), pred in zip(examples, predictions):
        is_valid = pred in count_tokens
        confusion[gold][pred if is_valid else 'other'] += 1
        per_count[gold][0] += pred == gold
        per_count[gold][1] += 1
        if not is_valid:
            invalid += 1
        elif int(pred) < int(gold):
            under += 1
        elif int(pred) > int(gold):
            over += 1

    # Scratchpad states written by the model itself (free-running).
    allowed_states = set(allowed_state_tokens(condition, count_tokens))
    state_correct = state_total = invalid_states = trace_exact = 0
    x_step_correct = x_step_total = follows_trace = 0
    for (body, gold), pred, trace in zip(examples, predictions, traces):
        gold_states = gold_state_tokens(body, condition)
        if not gold_states:
            continue
        matches = [written == expected for written, expected in zip(trace, gold_states)]
        state_correct += sum(matches)
        state_total += len(gold_states)
        invalid_states += sum(written not in allowed_states for written in trace)
        trace_exact += all(matches)
        for index, char in enumerate(body):
            if char == 'X':
                x_step_correct += matches[index]
                x_step_total += 1
        follows_trace += pred == trace[-1]

    nonzero = [gold for gold in count_tokens if gold != '0']
    count = len(examples)
    has_states = state_total > 0
    metrics = {
        'answer_loss': answer_loss,
        'acc_all': safe_divide(sum(c for c, _ in per_count.values()), count),
        'acc_k1plus': safe_divide(
            sum(per_count[gold][0] for gold in nonzero),
            sum(per_count[gold][1] for gold in nonzero),
        ),
        **{f'acc_k{gold}': safe_divide(*per_count[gold]) for gold in count_tokens},
        'under_rate': safe_divide(under, count),
        'over_rate': safe_divide(over, count),
        'invalid_rate': safe_divide(invalid, count),
        'state_acc': safe_divide(state_correct, state_total),
        'state_acc_at_x': safe_divide(x_step_correct, x_step_total),
        'trace_exact_acc': safe_divide(trace_exact, count) if has_states else None,
        'invalid_state_rate': safe_divide(invalid_states, state_total),
        'answer_follows_trace': (
            safe_divide(follows_trace, count)
            if has_states and condition == 'meaningful' else None
        ),
    }
    return metrics, confusion


def format_metric(value):
    return 'n/a' if value is None else f"{value:.2%}"


def print_split(name, metrics, confusion, count):
    print(f"\n{name} ({count} examples)")
    print(
        f"  answer loss {metrics['answer_loss']:.4f} | "
        f"acc all {format_metric(metrics['acc_all'])} | "
        f"acc k=1-3 {format_metric(metrics['acc_k1plus'])}"
    )
    print("  per count: " + " | ".join(
        f"k={gold} {format_metric(metrics[f'acc_k{gold}'])}" for gold in count_tokens
    ))
    print(
        f"  errors: under-count {format_metric(metrics['under_rate'])} | "
        f"over-count {format_metric(metrics['over_rate'])} | "
        f"invalid {format_metric(metrics['invalid_rate'])}"
    )
    if metrics['state_acc'] is not None:
        print(
            f"  written states: acc {format_metric(metrics['state_acc'])} | "
            f"right after X {format_metric(metrics['state_acc_at_x'])} | "
            f"whole trace {format_metric(metrics['trace_exact_acc'])} | "
            f"invalid {format_metric(metrics['invalid_state_rate'])} | "
            f"answer = last state {format_metric(metrics['answer_follows_trace'])}"
        )
    columns = count_tokens + ['other']
    print("  confusion (rows = gold, columns = predicted):")
    print("          " + "".join(f"{column:>7}" for column in columns))
    for gold in count_tokens:
        print(f"    k={gold}  " + "".join(f"{confusion[gold][c]:>7}" for c in columns))


def confusion_string(confusion):
    columns = count_tokens + ['other']
    return ';'.join(
        f"{gold}:" + ','.join(str(confusion[gold][c]) for c in columns)
        for gold in count_tokens
    )


val_examples = load_raw_examples(data_dir, 'val', meta)
test_examples = load_raw_examples(data_dir, 'test', meta)

val_predictions, val_traces, val_loss = predict(val_examples)
test_predictions, test_traces, test_loss = predict(test_examples)
val_metrics, val_confusion = split_metrics(
    val_examples, val_predictions, val_traces, val_loss
)
test_metrics, test_confusion = split_metrics(
    test_examples, test_predictions, test_traces, test_loss
)

print("=== Counting Evaluation ===")
print(
    f"condition={condition} | pos_type={pos_type} | seed={train_seed} "
    f"| iteration={checkpoint_iteration} | data_seed={meta['data_seed']} "
    f"| length={meta['length']} | seen X positions 0-{meta['seen_end'] - 1}"
)
print_split('seen-position validation', val_metrics, val_confusion, len(val_examples))
print_split('unseen-position test', test_metrics, test_confusion, len(test_examples))


def append_csv(path, header, rows):
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        if fresh:
            writer.writerow(header)
        writer.writerows(rows)


def csv_value(value):
    return '' if value is None else f"{value:.6f}"


if log_results:
    metric_names = list(val_metrics)
    append_csv(
        results_csv,
        ['timestamp', 'condition', 'pos_type', 'seed', 'data_seed', 'length',
         'seen_end', 'iteration']
        + [f'val_{name}' for name in metric_names]
        + [f'heldout_{name}' for name in metric_names]
        + ['val_confusion', 'heldout_confusion'],
        [[
            datetime.now().isoformat(timespec='seconds'), condition, pos_type,
            train_seed, meta['data_seed'], meta['length'], meta['seen_end'],
            checkpoint_iteration,
            *[csv_value(val_metrics[name]) for name in metric_names],
            *[csv_value(test_metrics[name]) for name in metric_names],
            confusion_string(val_confusion), confusion_string(test_confusion),
        ]],
    )

    if log_predictions:
        prediction_rows = []
        for split_name, examples, predictions, traces in (
            ('val', val_examples, val_predictions, val_traces),
            ('test', test_examples, test_predictions, test_traces),
        ):
            for (body, gold), pred, trace in zip(examples, predictions, traces):
                positions = ' '.join(str(i) for i, char in enumerate(body) if char == 'X')
                prediction_rows.append([
                    condition, pos_type, train_seed, meta['data_seed'],
                    checkpoint_iteration, split_name, body, positions, gold, pred,
                    int(pred == gold), trace,
                ])
        append_csv(
            predictions_csv,
            ['condition', 'pos_type', 'seed', 'data_seed', 'iteration', 'split',
             'body', 'x_positions', 'gold', 'pred', 'correct', 'pred_states'],
            prediction_rows,
        )
        print(f"\nlogged -> {results_csv} and {predictions_csv}")
    else:
        print(f"\nlogged -> {results_csv}")
