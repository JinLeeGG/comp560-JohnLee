"""
Evaluation script for the detection task.

Reads the held-out test examples (data/detect/test.txt), feeds each '<20 digits>:'
prompt to the model, generates a SINGLE token, and checks it against the gold Y/N label.

Because the task is binary, chance is 50%. We report per-class accuracy (Y vs N), not
just overall, so we can see e.g. "good at detecting X, bad at confirming its absence".
For errors we print where the X was, which is what we care about in the generalization
experiment (do failures cluster at the held-out positions?).

Usage:
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u evaluate.py config/basic.py
    # point at a different test set:
    ... python -u evaluate.py config/basic.py --test_file=data/detect/test.txt
"""
import os
import sys
import pickle
import torch

# add nanoGPT to path so we can import the model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'comp560-nanoGPT'))
from model import GPTConfig, GPT

# ----- config (can be overridden by configurator.py) -----
out_dir = 'out'
data_dir = 'data/detect'
test_file = ''          # if empty, defaults to <data_dir>/test.txt
device = 'cpu'
seed = 1337
num_test = 0            # 0 = use all examples in the test file
show_errors = 10
# ---------------------------------------------------------

# load configurator overrides (config file + --key=value args)
exec(open(os.environ.get('NANOGPT_CONFIG', 'configurator.py')).read())

if not test_file:
    test_file = os.path.join(data_dir, 'test.txt')

torch.manual_seed(seed)

# load meta (vocab)
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return ''.join(itos[i] for i in ids)

# load model from checkpoint
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# load test examples
with open(test_file) as f:
    examples = [line for line in f.read().splitlines() if line]
if num_test:
    examples = examples[:num_test]

# evaluate: one generated token per example, compared to the gold label
counts = {'Y': [0, 0], 'N': [0, 0]}   # label -> [correct, total]
nonbinary = 0
errors = []

for e in examples:
    body, gold = e.split(':')
    prompt = body + ':'
    prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    with torch.no_grad():
        out = model.generate(prompt_ids, max_new_tokens=1, temperature=0.1, top_k=1)
    pred = decode([out[0, -1].item()])

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
print(f"examples: {total}   (chance = 50%)")
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
