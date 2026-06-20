"""
Evaluation script for the relative-order task (from-scratch micro-transformer engine).

Reads the held-out test examples (data/order/test.txt), feeds each '<20 chars>:' prompt to
the model, reads the argmax at the answer position (the token after ':'), and checks it
against the gold T/F label.

    LABEL MAPPING (must match prepare.py):  T  iff  index(X) < index(Y).

Because the task is binary, chance is 50%. We report PER-CLASS accuracy (T vs F), not just
overall: a model that always outputs one label scores 50% overall, and per-class accuracy
is what makes that failure mode visible. For errors we print where X and Y landed.

This script also LOGS results so figures regenerate themselves as methods/seeds accumulate:
  - results.csv     : one aggregate row per train+eval run (drives the bar chart).
  - predictions.csv : one row per example of a DIAGNOSTIC SWEEP that spans ALL position
                      pairs (train region AND held-out region), so plot.py can show the
                      per-position "cliff". The sweep is generated here, deterministically,
                      independent of the data split, so heatmaps are comparable across runs.

Usage (from generalization-order/):
    ../venv/bin/python evaluate.py config/basic.py --pos_type=none
    ../venv/bin/python evaluate.py config/basic.py --test_file=data/order/test.txt
"""
import os
import sys
import csv
import pickle
import random
from ast import literal_eval
from datetime import datetime

import torch

from model import MicroTransformer, MicroTransformerConfig

# ----- config (overridable by the config file + --key=value args) -----
out_dir = 'out'
data_dir = 'data/order'
test_file = ''          # if empty, defaults to <data_dir>/test.txt
device = 'cpu'
seed = 1337
num_test = 0            # 0 = use all examples in the test file
show_errors = 10
results_csv = 'results.csv'        # aggregate, one row per run
predictions_csv = 'predictions.csv'  # per-example diagnostic sweep
sweep_per_pair = 10     # examples per (x_pos, y_pos) cell in the diagnostic sweep
log_results = True      # set --log_results=False to evaluate without touching the CSVs
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

# load meta (vocab + split info, written by prepare.py)
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
split = meta.get('split', 'unknown')
split_detail = meta.get('split_detail', '')
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join(itos[i] for i in ids)
LENGTH = 20

# load model from checkpoint (model_args -- including pos_type -- travel with the ckpt;
# so do the training seed and the in-distribution val accuracy)
checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device, weights_only=False)
model = MicroTransformer(MicroTransformerConfig(**checkpoint['model_args']))
model.load_state_dict(checkpoint['model'])
model.eval()
model.to(device)
pos_type = checkpoint['model_args'].get('pos_type')
train_seed = checkpoint.get('seed', seed)     # authoritative seed = the one trained with
val_acc = checkpoint.get('val_acc')           # in-distribution val accuracy (overall)


@torch.no_grad()
def predict_batch(bodies):
    """Greedy argmax label for a list of length-LENGTH bodies (feeds body+':')."""
    ids = torch.tensor([encode(b + ':') for b in bodies], dtype=torch.long, device=device)
    preds = []
    for i in range(0, len(ids), 512):
        logits = model(ids[i:i + 512])[:, -1, :]
        preds.extend(decode([t]) for t in logits.argmax(dim=-1).tolist())
    return preds


# ---------------- held-out test set (drives heldout_acc in the bar chart) ----------------
with open(test_file) as f:
    examples = [line for line in f.read().splitlines() if line]
if num_test:
    examples = examples[:num_test]

counts = {'T': [0, 0], 'F': [0, 0]}   # label -> [correct, total]
nonbinary = 0
errors = []
bodies = [e.split(':')[0] for e in examples]
golds = [e.split(':')[1] for e in examples]
preds = predict_batch(bodies)
for body, gold, pred in zip(bodies, golds, preds):
    counts[gold][1] += 1
    if pred == gold:
        counts[gold][0] += 1
    else:
        errors.append((body, gold, pred))
    if pred not in ('T', 'F'):
        nonbinary += 1

cT, tT = counts['T']
cF, tF = counts['F']
correct, total = cT + cF, tT + tF
heldout_acc = correct / total if total else 0.0
heldout_T_acc = cT / tT if tT else 0.0
heldout_F_acc = cF / tF if tF else 0.0

# report
print("=== Relative-order Evaluation ===")
print(f"test_file: {test_file}")
print(f"pos_type : {pos_type} | split: {split} ({split_detail}) | seed: {train_seed}")
print(f"val acc  : {val_acc:.2%}" if val_acc is not None else "val acc  : (not in ckpt)")
print(f"examples : {total}   (chance = 50%)")
if tT:
    print(f"T (X before Y): {cT}/{tT} = {heldout_T_acc:.2%}")
if tF:
    print(f"F (Y before X): {cF}/{tF} = {heldout_F_acc:.2%}")
if total:
    print(f"OVERALL       : {correct}/{total} = {heldout_acc:.2%}")
if nonbinary:
    print(f"(note: {nonbinary} predictions were neither T nor F)")
if errors and show_errors:
    print(f"\nfirst {min(show_errors, len(errors))} errors (X/Y positions):")
    for body, gold, pred in errors[:show_errors]:
        print(f"  X@{body.index('X'):>2} Y@{body.index('Y'):>2}  gold={gold} pred={pred}   {body}")


# ----------------------------- CSV logging -----------------------------
def append_csv(path, header, rows):
    """Append rows to a plain CSV, writing the header first only if the file is new/empty."""
    fresh = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(header)
        w.writerows(rows)


def build_sweep(per_pair, length, sweep_seed=0):
    """Diagnostic sweep: for EVERY ordered position pair (x_pos != y_pos), `per_pair` random
    bodies with X at x_pos and Y at y_pos. Spans the whole position range (both the train
    and held-out regions) and is class-balanced (x<y -> T, x>y -> F). Deterministic, so the
    per-position heatmap is comparable across runs/methods."""
    rng = random.Random(sweep_seed)
    out = []
    for xp in range(length):
        for yp in range(length):
            if xp == yp:
                continue
            label = 'T' if xp < yp else 'F'
            for _ in range(per_pair):
                chars = [rng.choice('0123456789') for _ in range(length)]
                chars[xp], chars[yp] = 'X', 'Y'
                out.append((''.join(chars), xp, yp, label))
    return out


if log_results:
    # 1) aggregate row -> results.csv
    append_csv(
        results_csv,
        ['timestamp', 'pos_type', 'split', 'split_detail', 'seed',
         'val_acc', 'heldout_acc', 'heldout_T_acc', 'heldout_F_acc'],
        [[datetime.now().isoformat(timespec='seconds'), pos_type, split, split_detail,
          train_seed,
          f'{val_acc:.6f}' if val_acc is not None else '',
          f'{heldout_acc:.6f}', f'{heldout_T_acc:.6f}', f'{heldout_F_acc:.6f}']])

    # 2) per-example diagnostic sweep -> predictions.csv
    sweep = build_sweep(sweep_per_pair, LENGTH)
    sweep_preds = predict_batch([b for b, _, _, _ in sweep])
    rows = [[pos_type, split, train_seed, xp, yp, gold, pred, int(pred == gold)]
            for (_, xp, yp, gold), pred in zip(sweep, sweep_preds)]
    append_csv(predictions_csv,
               ['pos_type', 'split', 'seed', 'x_pos', 'y_pos', 'gold', 'pred', 'correct'],
               rows)

    sweep_acc = sum(r[7] for r in rows) / len(rows)
    print(f"\nlogged -> {results_csv} (1 row) and {predictions_csv} "
          f"({len(rows)} sweep rows, sweep acc {sweep_acc:.2%})")
