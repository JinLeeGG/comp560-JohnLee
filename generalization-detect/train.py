"""
Training loop for the from-scratch micro-transformer (detection task).

Two deliberate departures from the nanoGPT flat-stream baseline (see the spec):

  1. Per-example batching. Each row is exactly one example -- "<20 digits>:<label>" --
     so there is no packing and no truncation, and an example's positions never blur
     across a training window. (Critical once position determines the answer.)

  2. Answer-token-only loss. Cross-entropy is computed ONLY at the answer position
     (the token after ':'), not over every token. The 20 digits are random and
     unpredictable, so supervising them is wasted signal that flattens the loss curve;
     supervising just the answer makes the loss meaningful and training cleaner.

The detection task is easy enough that a correct engine still hits ~100%, so this stays a
valid sanity check while being cleaner than the baseline.

Usage (from generalization-detect/):
    ../venv/bin/python train.py config/basic.py
    ../venv/bin/python train.py config/basic.py --max_iters=500 --pos_type=none
"""
import os
import sys
import math
import time
import pickle
from ast import literal_eval

import numpy as np
import torch
from torch.nn import functional as F

from model import MicroTransformer, MicroTransformerConfig

# ---------------------------- config (overridable) ----------------------------
out_dir = 'out'
data_dir = 'data/detect'
eval_interval = 250
log_interval = 100

# model -- only pos_type is meant to vary across the PE comparison
n_layer = 4
n_head = 4
n_embd = 128
block_size = 64
dropout = 0.0
bias = True
pos_type = 'learned'

# optimizer / schedule (in line with the previous nanoGPT config)
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
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join(itos[i] for i in ids)


def load_examples(name):
    """Read a flat .bin stream (written by prepare.py) back into one example per row.

    Each example is "<digits>:<label>"; we feed the digits+':' as input and supervise the
    single label token. The model sees positions 0..T-2 and predicts the label at the
    last position, so the loss lands exactly on the answer token.
    """
    ids = np.fromfile(os.path.join(data_dir, f'{name}.bin'), dtype=np.uint16)
    lines = [ln for ln in decode(ids.tolist()).split('\n') if ln]
    toks = [encode(ln) for ln in lines]
    L = len(toks[0])
    assert all(len(t) == L for t in toks), \
        "examples are not all the same length -- per-example batching needs fixed length"
    arr = torch.tensor(toks, dtype=torch.long)
    x = arr[:, :-1]            # (N, T)   digits + ':'
    y = arr[:, -1]             # (N,)     the answer (label) token
    return x, y


Xtr, Ytr = load_examples('train')
Xval, Yval = load_examples('val')
print(f"data: train={len(Xtr)} val={len(Xval)} | seq_len={Xtr.size(1)} | vocab_size={vocab_size}")


def get_batch():
    ix = torch.randint(len(Xtr), (batch_size,))
    return Xtr[ix].to(device), Ytr[ix].to(device)


# ------------------------------- model ------------------------------
model_args = dict(vocab_size=vocab_size, block_size=block_size, n_layer=n_layer,
                  n_head=n_head, n_embd=n_embd, dropout=dropout, bias=bias, pos_type=pos_type)
model = MicroTransformer(MicroTransformerConfig(**model_args))
model.to(device)
print(f"model: pos_type={pos_type} | {model.num_params()/1e6:.2f}M params")

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
# Detection loss is uninformative (~random digits dominate the stream), so we checkpoint
# on best VAL ACCURACY, not loss.
best_val_acc = -1.0
t0 = time.time()
print(f"\ntraining for {max_iters} iters on {device}\n")

for it in range(max_iters + 1):
    lr = get_lr(it)
    for g in optimizer.param_groups:
        g['lr'] = lr

    if it % eval_interval == 0 or it == max_iters:
        val_loss, val_acc = evaluate(Xval, Yval)
        marker = ''
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save({'model': model.state_dict(), 'model_args': model_args,
                        'iter': it, 'val_acc': val_acc, 'val_loss': val_loss},
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

print(f"\ndone in {time.time() - t0:.1f}s | best val acc {best_val_acc:.2%} | ckpt -> {out_dir}/ckpt.pt")
