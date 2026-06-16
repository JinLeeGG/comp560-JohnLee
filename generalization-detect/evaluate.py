"""
Evaluation script for the detection task (from-scratch micro-transformer engine).

Reads the held-out test examples (data/detect/test.txt), feeds each '<20 digits>:'
prompt to the model, reads the argmax at the answer position (the token after ':'), and
checks it against the gold Y/N label.

Because the task is binary, chance is 50%. We report per-class accuracy (Y vs N), not
just overall, so we can see e.g. "good at detecting X, bad at confirming its absence".
For errors we print where the X was, which is what we care about in the generalization
experiment (do failures cluster at the held-out positions?).

Usage (from generalization-detect/):
    ../venv/bin/python evaluate.py config/basic.py
    # point at a different test set:
    ../venv/bin/python evaluate.py config/basic.py --test_file=data/detect/test.txt
"""
import os
import sys
import pickle
from ast import literal_eval

import torch

from model import MicroTransformer, MicroTransformerConfig

# ----- config (overridable by the config file + --key=value args) -----
out_dir = 'out'
data_dir = 'data/detect'
test_file = ''          # if empty, defaults to <data_dir>/test.txt
device = 'cpu'
seed = 1337
num_test = 0            # 0 = use all examples in the test file
show_errors = 10
# ----------------------------------------------------------------------

# poor-man's configurator: a bare arg is a config file to exec; --key=val overrides a key.
for arg in sys.argv[1:]:
    if '=' not in arg:
        assert not arg.startswith('--'), f"expected a config file, got {arg!r}"
        exec(open(arg).read())
    else:
        assert arg.startswith('--'), f"expected --key=value, got {arg!r}"
        key, val = arg[2:].split('=', 1)
        if key not in globals():
            continue                       # ignore train-only keys present in the config file
        try:
            val = literal_eval(val)
        except (SyntaxError, ValueError):
            pass
        globals()[key] = val

if not test_file:
    test_file = os.path.join(data_dir, 'test.txt')

torch.manual_seed(seed)

# load meta (vocab)
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join(itos[i] for i in ids)

# load model from checkpoint (model_args -- including pos_type -- travel with the ckpt)
checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device, weights_only=False)
model = MicroTransformer(MicroTransformerConfig(**checkpoint['model_args']))
model.load_state_dict(checkpoint['model'])
model.eval()
model.to(device)

# load test examples
with open(test_file) as f:
    examples = [line for line in f.read().splitlines() if line]
if num_test:
    examples = examples[:num_test]

# evaluate: argmax at the answer position, compared to the gold label
counts = {'Y': [0, 0], 'N': [0, 0]}   # label -> [correct, total]
nonbinary = 0
errors = []

for e in examples:
    body, gold = e.split(':')
    prompt_ids = torch.tensor([encode(body + ':')], dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model(prompt_ids)
    pred = decode([logits[0, -1].argmax().item()])

    counts[gold][1] += 1
    if pred == gold:
        counts[gold][0] += 1
    else:
        errors.append((body, gold, pred))
    if pred not in ('Y', 'N'):
        nonbinary += 1

# report
cY, tY = counts['Y']
cN, tN = counts['N']
correct, total = cY + cN, tY + tN

print("=== Detection Evaluation ===")
print(f"test_file: {test_file}")
print(f"pos_type : {checkpoint['model_args'].get('pos_type')}")
print(f"examples : {total}   (chance = 50%)")
if tY:
    print(f"Y (X present): {cY}/{tY} = {cY / tY:.2%}")
if tN:
    print(f"N (X absent) : {cN}/{tN} = {cN / tN:.2%}")
if total:
    print(f"OVERALL      : {correct}/{total} = {correct / total:.2%}")
if nonbinary:
    print(f"(note: {nonbinary} predictions were neither Y nor N)")

if errors and show_errors:
    print(f"\nfirst {min(show_errors, len(errors))} errors (X@position):")
    for body, gold, pred in errors[:show_errors]:
        xpos = body.index('X') if 'X' in body else '-'
        print(f"  X@{str(xpos):>2}  gold={gold} pred={pred}   {body}")
