"""
Training loop for the from-scratch micro-transformer (distance-1 task, background sweep).

Same engine and structure as generalization-distd / evenodd / order; only the data dir,
the model size, and the extra `b` bookkeeping differ. The two deliberate departures from the
nanoGPT flat-stream baseline carry over unchanged:

  1. Per-example batching. Each row is exactly one example -- "<LENGTH chars>:<label>" -- so
     there is no packing and no truncation, and an example's positions never blur across a
     training window. (Critical once position determines the answer.)

  2. Answer-token-only loss. Cross-entropy is computed ONLY at the answer position (the
     token after ':'), not over every token. The background slots are random and
     unpredictable, so supervising them is wasted signal -- and here that matters more than
     ever, since the background is the variable under study: at b=10 a per-token loss would
     be dominated by the un-learnable background and would drown out the answer signal.

Two additions specific to this experiment:

  - `b` (background diversity) and `length` are read from meta.pkl and STORED IN THE
    CHECKPOINT, so evaluate.py labels its results.csv row with the b the model was actually
    trained on rather than whatever data happens to sit on disk at eval time.
  - the val curve is appended to curves.csv (iter, val_loss, val_acc). Stage 1's gate is
    literally "watch the val curve" -- flat 50% = it cannot learn, climbing = good -- so the
    curve is recorded rather than left to scroll past in the terminal.
    (plot.py --curve draws it.)

Usage (from generalization-background/):
    ../venv/bin/python train.py config/basic.py --seed=1337
    ../venv/bin/python train.py config/basic.py --max_iters=500 --pos_type=rope
"""
import os
import sys
import csv
import math
import time
import pickle
from ast import literal_eval
from datetime import datetime

import numpy as np
import torch
from torch.nn import functional as F

from model import MicroTransformer, MicroTransformerConfig

# ---------------------------- config (overridable) ----------------------------
out_dir = 'out'
data_dir = 'data/background'
eval_interval = 100
log_interval = 100

# model -- pos_type is pinned to t5 for this study (the swept variable is `b`, in the data)
n_layer = 3
n_head = 2
n_embd = 32
block_size = 8
dropout = 0.0
bias = True
pos_type = 't5'
causal = True
t5_bias_mode = 'auto'

# optimizer / schedule (in line with the previous task folders)
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

curves_csv = 'curves.csv'   # val curve log (one row per eval point); '' disables
device = 'cpu'
seed = 1337
# ------------------------------------------------------------------------------

# poor-man's configurator: a bare arg is a config file to exec; --key=val overrides a key.
for arg in sys.argv[1:]:
    if '=' not in arg:
        assert not arg.startswith('--'), f"expected a config file, got {arg!r}"
        print(f"Overriding config with {arg}")
        exec(open(arg).read())
    else:
        assert arg.startswith('--'), f"expected --key=value, got {arg!r}"
        key, val = arg[2:].split('=', 1)
        assert key in globals(), f"unknown config key: {key}"
        try:
            val = literal_eval(val)
        except (SyntaxError, ValueError):
            pass
        assert type(val) == type(globals()[key]), \
            f"type mismatch for {key}: {type(val)} vs {type(globals()[key])}"
        globals()[key] = val
        print(f"Overriding: {key} = {val}")

torch.manual_seed(seed)
np.random.seed(seed)
os.makedirs(out_dir, exist_ok=True)

# ------------------------------- data -------------------------------
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
vocab_size = meta['vocab_size']
b = meta['b']                       # background diversity of the data on disk
length = meta['length']
split = meta.get('split', 'unknown')
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join(itos[i] for i in ids)


def load_examples(name):
    """Read a flat .bin stream (written by prepare.py) back into one example per row.

    Each example is "<body><label>" with no separator (see prepare.py); we feed the body as
    input and supervise the single label token. The model sees positions 0..T-2 (the LENGTH
    body tokens) and predicts the label at the last position, so the loss lands exactly on the
    answer token.
    """
    ids = np.fromfile(os.path.join(data_dir, f'{name}.bin'), dtype=np.uint16)
    lines = [ln for ln in decode(ids.tolist()).split('\n') if ln]
    toks = [encode(ln) for ln in lines]
    L = len(toks[0])
    assert all(len(t) == L for t in toks), \
        "examples are not all the same length -- per-example batching needs fixed length"
    arr = torch.tensor(toks, dtype=torch.long)
    x = arr[:, :-1]            # (N, T)   the LENGTH body tokens
    y = arr[:, -1]             # (N,)     the answer (label) token
    return x, y


Xtr, Ytr = load_examples('train')
Xval, Yval = load_examples('val')
seq_len = Xtr.size(1)
# Caught here rather than deep in the model: Stage 3 raises `length`, and block_size has to
# be raised with it. The message says exactly what to set.
assert seq_len <= block_size, (
    f"seq_len {seq_len} exceeds block_size {block_size} -- length={length} needs "
    f"block_size >= {seq_len} (set it in config/basic.py or pass --block_size={length + 2})")
print(f"data: train={len(Xtr)} val={len(Xval)} | seq_len={seq_len} | vocab_size={vocab_size} "
      f"| length={length} | b={b} | split={split}")


def get_batch():
    ix = torch.randint(len(Xtr), (batch_size,))
    return Xtr[ix].to(device), Ytr[ix].to(device)


# ------------------------------- model ------------------------------
model_args = dict(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                  n_head=n_head, n_embd=n_embd, dropout=dropout, bias=bias,
                  pos_type=pos_type, causal=causal, t5_bias_mode=t5_bias_mode)
model = MicroTransformer(MicroTransformerConfig(**model_args))
model.to(device)
# Parameter count is printed (and logged) on purpose: the whole point of the small model is
# that it must not be big enough to trivially solve the task at every b.
print(f"model: pos_type={pos_type} | causal={causal} | t5_bias_mode={t5_bias_mode} | "
      f"n_layer={n_layer} n_head={n_head} n_embd={n_embd} | "
      f"{model.num_params():,} params ({model.num_params()/1e6:.3f}M)")

# AdamW with the usual decay / no-decay split (no weight decay on biases & LayerNorm).
decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{'params': decay, 'weight_decay': weight_decay},
     {'params': no_decay, 'weight_decay': 0.0}],
    lr=learning_rate, betas=(beta1, beta2))


def get_lr(it):
    """Linear warmup then cosine decay to min_lr (matches the nanoGPT schedule)."""
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    if it > lr_decay_iters:
        return min_lr
    ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))
    return min_lr + coeff * (learning_rate - min_lr)


@torch.no_grad()
def evaluate(X, Y):
    """Answer-token loss and accuracy over a whole split (in minibatches)."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    for i in range(0, len(X), 512):
        xb, yb = X[i:i + 512].to(device), Y[i:i + 512].to(device)
        logits = model(xb)[:, -1, :]                       # logits at the answer position
        loss_sum += F.cross_entropy(logits, yb, reduction='sum').item()
        correct += (logits.argmax(dim=-1) == yb).sum().item()
        total += len(yb)
    model.train()
    return loss_sum / total, correct / total


# ------------------------------- train ------------------------------
# Checkpoint on best VAL ACCURACY (not loss): accuracy is the quantity we actually report,
# and it is a robust selector for this binary answer-token task.
best_val_acc = -1.0
curve = []                       # (iter, val_loss, val_acc) -> curves.csv
t0 = time.time()
print(f"\ntraining for {max_iters} iters on {device}\n")

for it in range(max_iters + 1):
    lr = get_lr(it)
    for g in optimizer.param_groups:
        g['lr'] = lr

    if it % eval_interval == 0 or it == max_iters:
        val_loss, val_acc = evaluate(Xval, Yval)
        curve.append((it, val_loss, val_acc))
        marker = ''
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({'model': model.state_dict(), 'model_args': model_args,
                        'iter': it, 'val_acc': val_acc, 'val_loss': val_loss, 'seed': seed,
                        'b': b, 'length': length, 'split': split,
                        'num_params': model.num_params()},
                       os.path.join(out_dir, 'ckpt.pt'))
            marker = '  (saved)'
        print(f"iter {it:>5}: val loss {val_loss:.4f} | val acc {val_acc:.2%}{marker}")

    if it == max_iters:
        break

    xb, yb = get_batch()
    logits = model(xb)
    loss = F.cross_entropy(logits[:, -1, :], yb)           # answer-token-only loss
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    if it % log_interval == 0:
        print(f"iter {it:>5}: train loss {loss.item():.4f} | lr {lr:.2e}")

print(f"\ndone in {time.time() - t0:.1f}s | best val acc {best_val_acc:.2%} | "
      f"ckpt -> {out_dir}/ckpt.pt")

# --- val curve -> curves.csv (Stage 1's gate; also shows whether max_iters is wasteful) ---
if curves_csv:
    stamp = datetime.now().isoformat(timespec='seconds')
    fresh = (not os.path.exists(curves_csv)) or os.path.getsize(curves_csv) == 0
    with open(curves_csv, 'a', newline='') as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(['timestamp', 'pos_type', 'split', 'length', 'b', 'seed',
                        'iter', 'val_loss', 'val_acc'])
        w.writerows([[stamp, pos_type, split, length, b, seed, it,
                      f'{vl:.6f}', f'{va:.6f}'] for it, vl, va in curve])
    print(f"val curve -> {curves_csv} ({len(curve)} points)")
