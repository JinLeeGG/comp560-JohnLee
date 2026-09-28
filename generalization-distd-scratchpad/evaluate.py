"""Teacher-forced and free-running evaluation for all scratchpad conditions."""
import csv
import os
import sys
from ast import literal_eval
from datetime import datetime

import torch
from torch.nn import functional as F

from dataset import load_meta, load_raw_examples, tensorize_examples
from engine import MicroTransformer, MicroTransformerConfig
from scratchpad import (
    IGNORE_INDEX,
    allowed_state_tokens,
    distance_and_label,
    gold_state_tokens,
)


# ---------------------------- config (overridable) ----------------------------
out_dir = 'out'
checkpoint_name = 'ckpt.pt'
data_dir = 'data/distd_scratchpad'
device = 'cpu'
seed = 1337
inference_batch_size = 256
num_val = 0                 # 0 = all
num_test = 0                # 0 = all
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

checkpoint = torch.load(
    os.path.join(out_dir, checkpoint_name), map_location=device, weights_only=False
)
condition = checkpoint['condition']
train_seed = checkpoint['seed']
checkpoint_iteration = checkpoint['iteration']
model = MicroTransformer(MicroTransformerConfig(**checkpoint['model_args']))
model.load_state_dict(checkpoint['model'])
model.to(device)
model.eval()


def decode_ids(ids):
    return [itos[int(token_id)] for token_id in ids]


def selected_logits(logits, targets):
    mask = targets != IGNORE_INDEX
    return logits[mask], targets[mask]


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else None


@torch.no_grad()
def teacher_forced_metrics(examples):
    split = tensorize_examples(examples, condition, meta)
    totals = {
        'state_loss': 0.0, 'state_correct': 0, 'state_count': 0,
        'answer_loss': 0.0, 'answer_correct': 0, 'answer_count': 0,
    }
    for start in range(0, len(split.inputs), 512):
        inputs = split.inputs[start:start + 512].to(device)
        state_targets = split.state_targets[start:start + 512].to(device)
        answer_targets = split.answer_targets[start:start + 512].to(device)
        logits = model(inputs)

        state_logits, state_gold = selected_logits(logits, state_targets)
        if len(state_gold):
            totals['state_loss'] += F.cross_entropy(
                state_logits, state_gold, reduction='sum'
            ).item()
            totals['state_correct'] += (state_logits.argmax(-1) == state_gold).sum().item()
            totals['state_count'] += len(state_gold)

        answer_logits, answer_gold = selected_logits(logits, answer_targets)
        totals['answer_loss'] += F.cross_entropy(
            answer_logits, answer_gold, reduction='sum'
        ).item()
        totals['answer_correct'] += (answer_logits.argmax(-1) == answer_gold).sum().item()
        totals['answer_count'] += len(answer_gold)

    state_count = totals['state_count']
    return {
        'teacher_state_loss': totals['state_loss'] / state_count if state_count else None,
        'teacher_state_acc': totals['state_correct'] / state_count if state_count else None,
        'teacher_answer_loss': totals['answer_loss'] / totals['answer_count'],
        'teacher_answer_acc': totals['answer_correct'] / totals['answer_count'],
    }


@torch.no_grad()
def predict_free_running(examples):
    """Return greedy final answers and generated state traces."""
    all_answers = []
    all_states = []

    for start in range(0, len(examples), inference_batch_size):
        batch = examples[start:start + inference_batch_size]
        bodies = [body for body, _ in batch]

        if condition == 'no_scratchpad':
            contexts = torch.tensor(
                [[stoi[token] for token in body + ':'] for body in bodies],
                dtype=torch.long,
                device=device,
            )
            answer_ids = model(contexts)[:, -1, :].argmax(-1)
            all_answers.extend(decode_ids(answer_ids.tolist()))
            all_states.extend([[] for _ in batch])
            continue

        contexts = torch.empty((len(batch), 0), dtype=torch.long, device=device)
        state_columns = []
        raw_prompts = [body + ':' for body in bodies]

        for raw_index in range(meta['length'] + 1):
            raw_ids = torch.tensor(
                [stoi[prompt[raw_index]] for prompt in raw_prompts],
                dtype=torch.long,
                device=device,
            ).unsqueeze(1)
            contexts = torch.cat((contexts, raw_ids), dim=1)
            state_ids = model(contexts)[:, -1, :].argmax(-1)
            state_columns.append(state_ids.cpu())
            contexts = torch.cat((contexts, state_ids.unsqueeze(1)), dim=1)

        answer_ids = model(contexts)[:, -1, :].argmax(-1)
        all_answers.extend(decode_ids(answer_ids.tolist()))
        state_matrix = torch.stack(state_columns, dim=1).tolist()
        all_states.extend([decode_ids(row) for row in state_matrix])

    return all_answers, all_states


def free_running_metrics(examples, answer_predictions, state_predictions):
    answer_correct = 0
    class_counts = {'T': [0, 0], 'F': [0, 0]}
    rows = []

    state_correct = 0
    state_count = 0
    trace_correct = 0
    invalid_count = 0
    decision_correct = 0
    decision_count = 0
    decision_by_class = {'T': [0, 0], 'F': [0, 0]}
    allowed_states = set(allowed_state_tokens(condition))

    for (body, label), answer_pred, states_pred in zip(
        examples, answer_predictions, state_predictions
    ):
        is_answer_correct = answer_pred == label
        answer_correct += is_answer_correct
        class_counts[label][0] += is_answer_correct
        class_counts[label][1] += 1

        gold_states = gold_state_tokens(body, condition, meta['threshold'])
        trace_match = None
        decision_pred = ''
        if gold_states:
            comparisons = [pred == gold for pred, gold in zip(states_pred, gold_states)]
            state_correct += sum(comparisons)
            state_count += len(gold_states)
            trace_match = all(comparisons)
            trace_correct += trace_match
            invalid_count += sum(state not in allowed_states for state in states_pred)

            if condition == 'meaningful':
                second_x = body.rfind('X')
                decision_pred = states_pred[second_x]
                expected_decision = label.lower()
                is_decision_correct = decision_pred == expected_decision
                decision_correct += is_decision_correct
                decision_count += 1
                decision_by_class[label][0] += is_decision_correct
                decision_by_class[label][1] += 1

        distance, _ = distance_and_label(body, meta['threshold'])
        rows.append({
            'body': body,
            'distance': distance,
            'gold': label,
            'answer_pred': answer_pred,
            'answer_correct': int(is_answer_correct),
            'decision_pred': decision_pred,
            'trace_exact': '' if trace_match is None else int(trace_match),
        })

    count = len(examples)
    metrics = {
        'free_answer_acc': safe_divide(answer_correct, count),
        'free_T_acc': safe_divide(class_counts['T'][0], class_counts['T'][1]),
        'free_F_acc': safe_divide(class_counts['F'][0], class_counts['F'][1]),
        'free_state_acc': safe_divide(state_correct, state_count),
        'trace_exact_acc': safe_divide(trace_correct, count) if state_count else None,
        'invalid_state_rate': safe_divide(invalid_count, state_count),
        'decision_acc': safe_divide(decision_correct, decision_count),
        'decision_T_acc': safe_divide(
            decision_by_class['T'][0], decision_by_class['T'][1]
        ),
        'decision_F_acc': safe_divide(
            decision_by_class['F'][0], decision_by_class['F'][1]
        ),
    }
    return metrics, rows


def evaluate_split(name, examples):
    teacher = teacher_forced_metrics(examples)
    answers, states = predict_free_running(examples)
    free, rows = free_running_metrics(examples, answers, states)
    return {**teacher, **free}, rows


def format_metric(value):
    return 'n/a' if value is None else f"{value:.2%}"


def format_loss(value):
    return 'n/a' if value is None else f"{value:.4f}"


def print_split(name, metrics, count):
    print(f"\n{name} ({count} examples)")
    print(
        f"  loss  : answer {format_loss(metrics['teacher_answer_loss'])} | "
        f"state {format_loss(metrics['teacher_state_loss'])}"
    )
    print(
        f"  answer: teacher {format_metric(metrics['teacher_answer_acc'])} | "
        f"free {format_metric(metrics['free_answer_acc'])} | "
        f"T {format_metric(metrics['free_T_acc'])} | "
        f"F {format_metric(metrics['free_F_acc'])}"
    )
    print(
        f"  state : teacher {format_metric(metrics['teacher_state_acc'])} | "
        f"free {format_metric(metrics['free_state_acc'])} | "
        f"trace {format_metric(metrics['trace_exact_acc'])} | "
        f"invalid {format_metric(metrics['invalid_state_rate'])}"
    )
    print(
        f"  decision: all {format_metric(metrics['decision_acc'])} | "
        f"T {format_metric(metrics['decision_T_acc'])} | "
        f"F {format_metric(metrics['decision_F_acc'])}"
    )


val_examples = load_raw_examples(data_dir, 'val', meta)
test_examples = load_raw_examples(data_dir, 'test', meta)
if num_val:
    val_examples = val_examples[:num_val]
if num_test:
    test_examples = test_examples[:num_test]

val_metrics, val_rows = evaluate_split('val', val_examples)
test_metrics, test_rows = evaluate_split('test', test_examples)

print("=== Scratchpad Distance Evaluation ===")
print(
    f"condition={condition} | seed={train_seed} | iteration={checkpoint_iteration} "
    f"| pos_type="
    f"{checkpoint['model_args']['pos_type']} | data_seed={meta['data_seed']}"
)
print_split('in-distribution validation', val_metrics, len(val_examples))
print_split('held-out test', test_metrics, len(test_examples))


def append_csv(path, header, rows):
    fresh = not os.path.exists(path) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as file:
        writer = csv.writer(file)
        if fresh:
            writer.writerow(header)
        writer.writerows(rows)


if log_results:
    metric_names = [
        'teacher_answer_loss', 'teacher_state_loss', 'teacher_answer_acc',
        'free_answer_acc', 'free_T_acc', 'free_F_acc', 'teacher_state_acc',
        'free_state_acc', 'trace_exact_acc',
        'invalid_state_rate', 'decision_acc', 'decision_T_acc', 'decision_F_acc',
    ]

    def csv_value(value):
        return '' if value is None else f"{value:.6f}"

    append_csv(
        results_csv,
        ['timestamp', 'condition', 'seed', 'data_seed', 'iteration']
        + [f'val_{name}' for name in metric_names]
        + [f'heldout_{name}' for name in metric_names],
        [[
            datetime.now().isoformat(timespec='seconds'), condition, train_seed,
            meta['data_seed'], checkpoint_iteration,
            *[csv_value(val_metrics[name]) for name in metric_names],
            *[csv_value(test_metrics[name]) for name in metric_names],
        ]],
    )

    if log_predictions:
        prediction_header = [
            'condition', 'seed', 'data_seed', 'iteration', 'split', 'body',
            'distance', 'gold', 'answer_pred', 'answer_correct',
            'decision_pred', 'trace_exact',
        ]
        prediction_rows = []
        for split_name, rows in (('val', val_rows), ('test', test_rows)):
            for row in rows:
                prediction_rows.append([
                    condition, train_seed, meta['data_seed'], checkpoint_iteration,
                    split_name, row['body'], row['distance'], row['gold'],
                    row['answer_pred'], row['answer_correct'],
                    row['decision_pred'], row['trace_exact'],
                ])
        append_csv(predictions_csv, prediction_header, prediction_rows)
        print(f"\nlogged -> {results_csv} and {predictions_csv}")
    else:
        print(f"\nlogged -> {results_csv}")
