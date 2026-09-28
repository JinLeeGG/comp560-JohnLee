"""Train one scratchpad condition on the fixed-length dist>=5 task."""
import csv
import math
import os
import sys
import time
from ast import literal_eval

import numpy as np
import torch
from torch.nn import functional as F

from dataset import load_meta, load_tensor_split
from engine import MicroTransformer, MicroTransformerConfig
from scratchpad import CONDITIONS, IGNORE_INDEX


# ---------------------------- config (overridable) ----------------------------
out_dir = 'out'
data_dir = 'data/distd_scratchpad'
condition = 'meaningful'
eval_interval = 250
log_interval = 100
checkpoint_iters = (500, 1000, 1500, 2000)
history_filename = 'history.csv'

n_layer = 4
n_head = 4
n_embd = 128
block_size = 64
dropout = 0.0
bias = True
pos_type = 'none'
causal = True
t5_bias_mode = 'auto'

batch_size = 64
max_iters = 2000
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 100
lr_decay_iters = 2000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

device = 'cpu'
seed = 1337
# ------------------------------------------------------------------------------


for arg in sys.argv[1:]:
    if '=' not in arg:
        if arg.startswith('--'):
            raise ValueError(f"expected a config file, got {arg!r}")
        print(f"Overriding config with {arg}")
        exec(open(arg).read())
    else:
        if not arg.startswith('--'):
            raise ValueError(f"expected --key=value, got {arg!r}")
        key, value = arg[2:].split('=', 1)
        if key not in globals():
            raise KeyError(f"unknown config key: {key}")
        try:
            value = literal_eval(value)
        except (SyntaxError, ValueError):
            pass
        if type(value) is not type(globals()[key]):
            raise TypeError(f"type mismatch for {key}: {type(value)} vs {type(globals()[key])}")
        globals()[key] = value
        print(f"Overriding: {key} = {value}")

if condition not in CONDITIONS:
    raise ValueError(f"condition must be one of {CONDITIONS}, got {condition!r}")
checkpoint_iters = tuple(sorted(set(checkpoint_iters)))
invalid_checkpoints = [
    iteration for iteration in checkpoint_iters
    if iteration <= 0 or iteration > max_iters
]
if invalid_checkpoints:
    raise ValueError(
        f"checkpoint iterations must be in 1..{max_iters}: {invalid_checkpoints}"
    )

torch.manual_seed(seed)
np.random.seed(seed)
os.makedirs(out_dir, exist_ok=True)

meta = load_meta(data_dir)
train_split = load_tensor_split(data_dir, 'train', condition, meta)
val_split = load_tensor_split(data_dir, 'val', condition, meta)

sequence_length = train_split.inputs.size(1)
if sequence_length > block_size:
    raise ValueError(f"sequence length {sequence_length} exceeds block_size {block_size}")
print(
    f"data: train={len(train_split.inputs)} val={len(val_split.inputs)} "
    f"| condition={condition} | seq_len={sequence_length} | vocab={meta['vocab_size']}"
)


def get_batch():
    indices = torch.randint(len(train_split.inputs), (batch_size,))
    return (
        train_split.inputs[indices].to(device),
        train_split.state_targets[indices].to(device),
        train_split.answer_targets[indices].to(device),
    )


model_args = dict(
    vocab_size=meta['vocab_size'], block_size=block_size, n_layer=n_layer,
    n_head=n_head, n_embd=n_embd, dropout=dropout, bias=bias,
    pos_type=pos_type, causal=causal, t5_bias_mode=t5_bias_mode,
)
model = MicroTransformer(MicroTransformerConfig(**model_args)).to(device)
print(f"model: NoPE={pos_type == 'none'} | {model.num_params()/1e6:.3f}M params")

decay = [parameter for parameter in model.parameters() if parameter.dim() >= 2]
no_decay = [parameter for parameter in model.parameters() if parameter.dim() < 2]
optimizer = torch.optim.AdamW(
    [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0},
    ],
    lr=learning_rate,
    betas=(beta1, beta2),
)


def get_lr(iteration):
    if iteration < warmup_iters:
        return learning_rate * (iteration + 1) / (warmup_iters + 1)
    if iteration > lr_decay_iters:
        return min_lr
    ratio = (iteration - warmup_iters) / (lr_decay_iters - warmup_iters)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coefficient * (learning_rate - min_lr)


def selected_logits(logits, targets):
    mask = targets != IGNORE_INDEX
    return logits[mask], targets[mask]


def training_losses(logits, state_targets, answer_targets):
    answer_logits, answer_gold = selected_logits(logits, answer_targets)
    answer_loss = F.cross_entropy(answer_logits, answer_gold)
    state_logits, state_gold = selected_logits(logits, state_targets)
    if len(state_gold):
        state_loss = F.cross_entropy(state_logits, state_gold)
        total_loss = state_loss + answer_loss
    else:
        state_loss = logits.sum() * 0.0
        total_loss = answer_loss
    return total_loss, state_loss, answer_loss


@torch.no_grad()
def evaluate_teacher_forced(split):
    model.eval()
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

    model.train()
    state_count = totals['state_count']
    return {
        'state_loss': totals['state_loss'] / state_count if state_count else None,
        'state_acc': totals['state_correct'] / state_count if state_count else None,
        'answer_loss': totals['answer_loss'] / totals['answer_count'],
        'answer_acc': totals['answer_correct'] / totals['answer_count'],
    }


def format_optional(value, kind='float'):
    if value is None:
        return 'n/a'
    return f"{value:.2%}" if kind == 'percent' else f"{value:.4f}"


def checkpoint_state(iteration, val_metrics):
    return {
        'model': model.state_dict(),
        'model_args': model_args,
        'condition': condition,
        'seed': seed,
        'iteration': iteration,
        'val_metrics': val_metrics,
        'data_meta': {
            'threshold': meta['threshold'],
            'length': meta['length'],
            'data_seed': meta['data_seed'],
            'vocab_size': meta['vocab_size'],
        },
    }


def write_history(rows):
    path = os.path.join(out_dir, history_filename)
    fieldnames = [
        'condition', 'seed', 'data_seed', 'iteration', 'learning_rate',
        'elapsed_seconds', 'val_answer_loss', 'val_answer_acc',
        'val_state_loss', 'val_state_acc',
    ]
    with open(path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


start_time = time.time()
print(f"\ntraining condition={condition} for {max_iters} iterations on {device}\n")
final_val_metrics = None
history_rows = []

for iteration in range(max_iters + 1):
    lr = get_lr(iteration)
    for group in optimizer.param_groups:
        group['lr'] = lr

    if (
        iteration % eval_interval == 0
        or iteration in checkpoint_iters
        or iteration == max_iters
    ):
        final_val_metrics = evaluate_teacher_forced(val_split)
        history_rows.append({
            'condition': condition,
            'seed': seed,
            'data_seed': meta['data_seed'],
            'iteration': iteration,
            'learning_rate': f'{lr:.10g}',
            'elapsed_seconds': f'{time.time() - start_time:.3f}',
            'val_answer_loss': f"{final_val_metrics['answer_loss']:.10g}",
            'val_answer_acc': f"{final_val_metrics['answer_acc']:.10g}",
            'val_state_loss': (
                '' if final_val_metrics['state_loss'] is None
                else f"{final_val_metrics['state_loss']:.10g}"
            ),
            'val_state_acc': (
                '' if final_val_metrics['state_acc'] is None
                else f"{final_val_metrics['state_acc']:.10g}"
            ),
        })
        write_history(history_rows)
        print(
            f"iter {iteration:>5}: val answer loss {final_val_metrics['answer_loss']:.4f} "
            f"acc {final_val_metrics['answer_acc']:.2%} | state loss "
            f"{format_optional(final_val_metrics['state_loss'])} acc "
            f"{format_optional(final_val_metrics['state_acc'], 'percent')}"
        )

    if iteration in checkpoint_iters:
        checkpoint_path = os.path.join(out_dir, f'ckpt_iter{iteration:04d}.pt')
        torch.save(checkpoint_state(iteration, final_val_metrics), checkpoint_path)
        print(f"saved checkpoint -> {checkpoint_path}")

    if iteration == max_iters:
        final_checkpoint_path = os.path.join(out_dir, 'ckpt.pt')
        torch.save(
            checkpoint_state(iteration, final_val_metrics), final_checkpoint_path
        )
        break

    inputs, state_targets, answer_targets = get_batch()
    logits = model(inputs)
    total_loss, state_loss, answer_loss = training_losses(
        logits, state_targets, answer_targets
    )
    optimizer.zero_grad(set_to_none=True)
    total_loss.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    if iteration % log_interval == 0:
        print(
            f"iter {iteration:>5}: train total {total_loss.item():.4f} "
            f"state {state_loss.item():.4f} answer {answer_loss.item():.4f} "
            f"| lr {lr:.2e}"
        )

print(
    f"\ndone in {time.time() - start_time:.1f}s | checkpoint -> "
    f"{os.path.join(out_dir, 'ckpt.pt')}"
)
