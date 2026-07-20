"""
Evaluation script for the distance-1 task (from-scratch micro-transformer engine).

Reads the held-out test examples (data/background/test.txt), feeds each LENGTH-char body to
the model (no ':' marker), reads the argmax at the last position (the predicted next token),
and checks it against the gold T/F label.

    LABEL MAPPING (must match prepare.py):  T iff the X-Y distance is exactly 1; F for any
    larger distance. X is always to the left of Y.

Because the task is binary, chance is 50%. We report PER-CLASS accuracy (T = distance 1,
F = larger distance), not just overall: a model that always outputs one label scores 50%
overall, and per-class accuracy is what makes that failure mode visible.

SUCCESS HERE IS TWO NUMBERS, NOT ONE. Every row records both:
    val_acc     -- in-distribution: did the model LEARN the task on trained positions?
    heldout_acc -- held-out:        did it GENERALIZE to unseen positions?
Read together per b:
    val high + held-out high -> T5 fully succeeds at this b
    val high + held-out low  -> learned but did not generalize (background hurts GENERALIZATION)
    val low                  -> never learned it (background hurts LEARNING itself)

This script also LOGS results so figures regenerate themselves as b values and seeds
accumulate:
  - results.csv     : one aggregate row per train+eval run, carrying `b` and `length`
                      (drives the headline accuracy-vs-b figure).
  - predictions.csv : one row per example of a DIAGNOSTIC SWEEP spanning ALL (X,Y) position
                      pairs -- the train region AND the held-out region -- recording both
                      positions and their gap, so plot.py can show the per-position pattern.
                      The sweep is generated here, deterministically, independent of the data
                      split, so figures are comparable across runs.

    Note on the per-position heatmap: with only 3 configurations per half at length 6,
    "distance 1 vs larger" is perfectly correlated with a position lookup inside the training
    region, so a model could win there without ever comparing X to Y. GAP structure
    (accuracy constant along the diagonal, i.e. all gap-1 cells alike) = the model reads
    relative distance; accuracy tracking WHERE Y sits = a position shortcut. For T5 we
    EXPECT the former (its bias is relative and gap 1 is its finest bucket) -- verify,
    do not assume.

`b`, `length` and the split are taken from the CHECKPOINT when present (train.py stores
them), not from whatever data currently sits on disk, so a re-prepared data dir cannot
silently mislabel a results row. A mismatch is reported loudly.

Usage (from generalization-background/):
    ../venv/bin/python evaluate.py config/basic.py --seed=1337
    ../venv/bin/python evaluate.py config/basic.py --test_file=data/background/test.txt
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
data_dir = 'data/background'
test_file = ''          # if empty, defaults to <data_dir>/test.txt
device = 'cpu'
seed = 1337
num_test = 0            # 0 = use all examples in the test file
show_errors = 10
results_csv = 'results.csv'          # aggregate, one row per run
predictions_csv = 'predictions.csv'  # per-example diagnostic sweep
sweep_per_pair = 40     # examples per (x_pos, y_pos) cell in the diagnostic sweep
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

# load meta (vocab + split info + length + b, written by prepare.py)
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi, itos = meta['stoi'], meta['itos']
split = meta.get('split', 'unknown')
split_detail = meta.get('split_detail', '')
LENGTH = meta['length']
B = meta['b']
BACKGROUND = meta.get('background', '0123456789'[:B])
MARKER = meta.get('marker', '')      # '' (answer is next token) or ':' (colon version)
encode = lambda s: [stoi[c] for c in s]
decode = lambda ids: ''.join(itos[i] for i in ids)


def xy_positions(body):
    """(x_pos, y_pos) of the single X and single Y in a body string."""
    return body.index('X'), body.index('Y')


# load model from checkpoint (model_args -- including pos_type -- travel with the ckpt; so do
# the training seed, the in-distribution val accuracy, and the b/length trained on)
checkpoint = torch.load(os.path.join(out_dir, 'ckpt.pt'), map_location=device, weights_only=False)
model = MicroTransformer(MicroTransformerConfig(**checkpoint['model_args']))
model.load_state_dict(checkpoint['model'])
model.eval()
model.to(device)
pos_type = checkpoint['model_args'].get('pos_type')
causal = checkpoint['model_args'].get('causal', True)
t5_bias_mode = checkpoint['model_args'].get('t5_bias_mode', 'auto')
train_seed = checkpoint.get('seed', seed)     # authoritative seed = the one trained with
val_acc = checkpoint.get('val_acc')           # in-distribution val accuracy (overall)
num_params = checkpoint.get('num_params', model.num_params())

# The checkpoint is authoritative for b/length; the data on disk may since have been
# re-prepared for a different b. Mismatch = the test set does not match the model, so say so
# rather than quietly logging a row under the wrong b.
ckpt_b = checkpoint.get('b', B)
ckpt_length = checkpoint.get('length', LENGTH)
if (ckpt_b, ckpt_length) != (B, LENGTH):
    print(f"WARNING: checkpoint was trained on b={ckpt_b}, length={ckpt_length} but "
          f"{data_dir} currently holds b={B}, length={LENGTH}. Re-run prepare.py with "
          f"--b={ckpt_b} --length={ckpt_length} before evaluating, or retrain. "
          f"Logging under the CHECKPOINT's b/length.")
B, LENGTH = ckpt_b, ckpt_length


@torch.no_grad()
def predict_batch(bodies):
    """Greedy argmax label for a list of length-LENGTH bodies (feeds the body as-is; there is
    no ':' marker, so the answer is the next token after the LENGTH-token body)."""
    ids = torch.tensor([encode(b + MARKER) for b in bodies], dtype=torch.long, device=device)
    preds = []
    for i in range(0, len(ids), 512):
        logits = model(ids[i:i + 512])[:, -1, :]
        preds.extend(decode([t]) for t in logits.argmax(dim=-1).tolist())
    return preds


# ---------------- held-out test set (drives heldout_acc in the headline figure) ----------------
with open(test_file) as f:
    examples = [line for line in f.read().splitlines() if line]
if num_test:
    examples = examples[:num_test]

counts = {'T': [0, 0], 'F': [0, 0]}   # label -> [correct, total]  (T=distance 1, F=larger)
nonbinary = 0
errors = []
bodies = [e[:LENGTH] for e in examples]              # body = first LENGTH chars
golds = [e[LENGTH + len(MARKER):] for e in examples]  # answer = what follows the MARKER
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
print("=== distance-1 task Evaluation ===")
print(f"test_file: {test_file}")
print(f"pos_type : {pos_type} | causal: {causal} | t5_bias_mode: {t5_bias_mode} | "
      f"split: {split} ({split_detail}) | length={LENGTH} | b={B} | seed: {train_seed}")
print(f"model    : {num_params:,} params | background set {list(BACKGROUND)}")
print(f"val acc  : {val_acc:.2%}   (in-distribution -- did it LEARN?)"
      if val_acc is not None else "val acc  : (not in ckpt)")
print(f"examples : {total}   (chance = 50%)")
if tT:
    print(f"T (distance 1)     : {cT}/{tT} = {heldout_T_acc:.2%}")
if tF:
    print(f"F (distance > 1)   : {cF}/{tF} = {heldout_F_acc:.2%}")
if total:
    print(f"OVERALL (held-out) : {correct}/{total} = {heldout_acc:.2%}   (did it GENERALIZE?)")
if nonbinary:
    print(f"(note: {nonbinary} predictions were neither T nor F)")
if errors and show_errors:
    print(f"\nfirst {min(show_errors, len(errors))} errors (X/Y positions, gap):")
    for body, gold, pred in errors[:show_errors]:
        x, y = xy_positions(body)
        print(f"  X@{x:>2} Y@{y:>2}  gap={y - x:>2}  gold={gold} pred={pred}   {body}")


# ----------------------------- CSV logging -----------------------------
def append_csv(path, header, rows):
    """Append rows to a plain CSV, writing the header first only if the file is new/empty."""
    fresh = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, 'a', newline='') as f:
        w = csv.writer(f)
        if fresh:
            w.writerow(header)
        w.writerows(rows)


def build_sweep(per_pair, length, background, sweep_seed=0):
    """Diagnostic sweep: for EVERY ordered position pair (x < y), `per_pair` random bodies
    with X at x and Y at y. Spans the whole position range (train region AND held-out
    region), so the heatmap shows the pattern everywhere, not just where the test pool
    lives. Backgrounds are drawn from the SAME size-b set the model was trained on -- a
    sweep with b=10 noise would be off-distribution for a b=1 model and would measure the
    wrong thing.

    Not class-balanced (larger-distance pairs vastly outnumber distance-1 pairs), which is fine: every
    figure built from it conditions on (x, y) or on the gap, so overall balance is
    irrelevant. Deterministic, so figures are comparable across runs and seeds."""
    rng = random.Random(sweep_seed)
    out = []
    for x in range(length):
        for y in range(x + 1, length):
            label = 'T' if y == x + 1 else 'F'
            for _ in range(per_pair):
                chars = [rng.choice(background) for _ in range(length)]
                chars[x], chars[y] = 'X', 'Y'
                out.append((''.join(chars), x, y, y - x, label))
    return out


if log_results:
    # 1) aggregate row -> results.csv  (b and length are columns: they are the swept axes)
    append_csv(
        results_csv,
        ['timestamp', 'pos_type', 'split', 'split_detail', 'length', 'b', 'seed',
         'num_params', 'val_acc', 'heldout_acc', 'heldout_T_acc', 'heldout_F_acc'],
        [[datetime.now().isoformat(timespec='seconds'), pos_type, split, split_detail,
          LENGTH, B, train_seed, num_params,
          f'{val_acc:.6f}' if val_acc is not None else '',
          f'{heldout_acc:.6f}', f'{heldout_T_acc:.6f}', f'{heldout_F_acc:.6f}']])

    # 2) per-example diagnostic sweep -> predictions.csv (gap logged explicitly, though it is
    #    derivable from x_pos/y_pos, so the heatmap code needn't recompute it)
    sweep = build_sweep(sweep_per_pair, LENGTH, BACKGROUND)
    sweep_preds = predict_batch([body for body, _, _, _, _ in sweep])
    rows = [[pos_type, split, LENGTH, B, train_seed, x, y, gap, gold, pred, int(pred == gold)]
            for (_, x, y, gap, gold), pred in zip(sweep, sweep_preds)]
    append_csv(predictions_csv,
               ['pos_type', 'split', 'length', 'b', 'seed', 'x_pos', 'y_pos', 'gap',
                'gold', 'pred', 'correct'],
               rows)

    sweep_acc = sum(r[10] for r in rows) / len(rows)
    print(f"\nlogged -> {results_csv} (1 row) and {predictions_csv} "
          f"({len(rows)} sweep rows, sweep acc {sweep_acc:.2%})")
