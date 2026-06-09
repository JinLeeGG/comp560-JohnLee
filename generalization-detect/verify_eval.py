"""
Correctness check for the detection evaluator + model (independent of test.txt).

Three checks:
  1. Controlled causal probes — take a fixed 20-digit string, flip X in/out at chosen
     positions, and confirm the model's single-token answer flips N <-> Y accordingly.
     This is the real test: does the model actually detect X (including at held-out
     positions), and does the eval logic return the right token?
  2. test.txt label integrity — every line: length 20, and (X present) iff (label Y),
     with exactly one X when present. Catches mislabeled data that could fake accuracy.
  3. test.txt X-position coverage — where X actually lands in the test set.

Run from generalization-detect/:
    ../venv/bin/python verify_eval.py
"""
import os
import sys
import pickle
from collections import Counter
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'comp560-nanoGPT'))
from model import GPTConfig, GPT

out_dir = 'out'
data_dir = 'data/detect'
device = 'cpu'

with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join(itos[i] for i in ids)

# load model exactly like evaluate.py does
ckpt = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device, weights_only=False)
model = GPT(GPTConfig(**ckpt['model_args']))
sd = ckpt['model']
for k in list(sd.keys()):
    if k.startswith('_orig_mod.'):
        sd[k[len('_orig_mod.'):]] = sd.pop(k)
model.load_state_dict(sd)
model.eval()


def predict(body):
    """body = 20-char string; return the model's single token after ':'."""
    ids = torch.tensor([encode(body + ':')], dtype=torch.long)
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=1, temperature=0.1, top_k=1)
    return decode([out[0, -1].item()])


print("=== 1. Controlled causal probes (flip X presence) ===")
base = "01234567890123456789"   # 20 digits, no X
print(f"  no X             -> {predict(base)}    (expect N)")
ok = True
if predict(base) != 'N':
    ok = False
for p in [0, 5, 9, 10, 12, 15, 19]:
    s = base[:p] + 'X' + base[p + 1:]
    region = "trained " if p <= 9 else "HELD-OUT"
    ans = predict(s)
    flag = "" if ans == 'Y' else "  <-- WRONG"
    if ans != 'Y':
        ok = False
    print(f"  X@{p:>2} ({region}) -> {ans}    (expect Y){flag}   {s}")
print(f"  --> probes {'ALL PASS' if ok else 'FAILED'}")

print("\n=== 2. test.txt label integrity ===")
lines = [l for l in open(os.path.join(data_dir, 'test.txt')).read().splitlines() if l]
bad = 0
for e in lines:
    body, gold = e.split(':')
    has_x = 'X' in body
    if len(body) != 20:
        bad += 1
    elif has_x and gold != 'Y':
        bad += 1
    elif (not has_x) and gold != 'N':
        bad += 1
    elif has_x and body.count('X') != 1:
        bad += 1
print(f"  checked {len(lines)} lines, integrity violations: {bad}")

print("\n=== 3. test.txt X-position coverage (Y examples) ===")
pos = Counter(e.split(':')[0].index('X') for e in lines if e.endswith('Y'))
print("  X positions in test:", dict(sorted(pos.items())))
