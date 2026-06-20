"""
Sanity probe for the relative-order task: isolate ORDER as the only variable.

For each trial we draw a single random digit body and a single pair of positions
(i < j), then build TWO inputs that are byte-for-byte identical except for which
symbol sits where:

    T-version:  X at i, Y at j   (X before Y)  -> gold T
    F-version:  Y at i, X at j   (Y before X)  -> gold F

The fillers, the positions, everything else is held fixed. So the ONLY thing that
can make the model's prediction flip T<->F is the relative order of X and Y. If the
model flips correctly, it is genuinely computing order -- there is no content leak
(the bag of tokens is identical between the two versions).

This is the key check for the surprising `none` (NoPE) result: a causal decoder has
NO explicit positional encoding under pos_type='none', yet the causal mask itself is
an ordering signal. This probe shows whether the model exploits that.

Usage (from generalization-order/, after training a checkpoint):
    ../venv/bin/python verify_order.py
    ../venv/bin/python verify_order.py --n_trials=1000
"""
import os
import sys
import pickle
import random
from ast import literal_eval

import torch

from model import MicroTransformer, MicroTransformerConfig

out_dir = 'out'
data_dir = 'data/order'
device = 'cpu'
seed = 1337
n_trials = 500
length = 20
show_errors = 10

for arg in sys.argv[1:]:
    if '=' in arg and arg.startswith('--'):
        key, val = arg[2:].split('=', 1)
        if key in globals():
            try:
                val = literal_eval(val)
            except (SyntaxError, ValueError):
                pass
            globals()[key] = val

random.seed(seed)
torch.manual_seed(seed)

with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join(itos[i] for i in ids)

ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device, weights_only=False)
model = MicroTransformer(MicroTransformerConfig(**ckpt['model_args']))
model.load_state_dict(ckpt['model'])
model.eval().to(device)


def predict(body):
    ids = torch.tensor([encode(body + ':')], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(ids)
    return decode([logits[0, -1].argmax().item()])


flips_correct = 0          # both versions predicted correctly (T->T and F->F)
errors = []
for _ in range(n_trials):
    i, j = sorted(random.sample(range(length), 2))
    fillers = [random.choice('0123456789') for _ in range(length)]
    t_body = fillers[:]; t_body[i], t_body[j] = 'X', 'Y'     # X before Y -> T
    f_body = fillers[:]; f_body[i], f_body[j] = 'Y', 'X'     # Y before X -> F
    t_body, f_body = ''.join(t_body), ''.join(f_body)
    pt, pf = predict(t_body), predict(f_body)
    if pt == 'T' and pf == 'F':
        flips_correct += 1
    else:
        errors.append((i, j, pt, pf))

print("=== Relative-order PROBE (order is the only variable) ===")
print(f"pos_type : {ckpt['model_args'].get('pos_type')}")
print(f"trials   : {n_trials}  (each = one matched T/F pair at fixed positions)")
print(f"both-correct flips: {flips_correct}/{n_trials} = {flips_correct / n_trials:.2%}")
print("(100% here means the model genuinely reads X-before-Y vs Y-before-X;")
print(" the bag of tokens is identical between the two versions, so this can't be a content leak.)")
if errors and show_errors:
    print(f"\nfirst {min(show_errors, len(errors))} mismatched pairs (i<j, pred on T-ver, pred on F-ver):")
    for i, j, pt, pf in errors[:show_errors]:
        print(f"  i={i:>2} j={j:>2}  T-version->{pt}  F-version->{pf}")
