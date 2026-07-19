"""
Distance-1 task data generator, with BACKGROUND DIVERSITY (b) as the swept knob.

Task: a fixed-length string containing X exactly once and Y exactly once, with X ALWAYS to
the left of Y (never Y...X). Every other slot is a background token. The model reads the
string, sees ':', and must output a SINGLE token:

    LABEL MAPPING (fixed -- never flip this):
        T  if  Y sits immediately after X   (distance 1)
        F  otherwise                        (something sits between them, gap > 1)

    0XY000:T   <- X@1 Y@2  distance 1 -> T
    X00Y00:F   <- X@0 Y@3  gap 3  -> a gap    -> F

Since X is always before Y, the label is purely "is the distance 1, or larger".
The task is deliberately EASY: it is not meant to be the bottleneck -- the background is.

Why this task exists (the point of the whole experiment)
-------------------------------------------------------
Across the earlier tasks (order / evenodd / distd) T5 kept underperforming and it was never
clear why. Every one of those tasks used a BUSY background: filler slots were random digits
0-9, i.e. b=10. Hypothesis: T5 was not failing at the task, it was failing to FIND X and Y
inside all that digit noise. This generator isolates that by making the background diversity
an explicit knob:

    b = the number of distinct background token types
        b=1  -> {0}          (background is a constant wall of '0')
        b=2  -> {0,1}
        ...
        b=10 -> {0,...,9}    (the busy background every earlier task used)

Background slots are filled uniformly at random from that size-b set. Everything else --
the task, the length, the split, the X-before-Y rule -- is held fixed as b varies, so any
change in accuracy across b is attributable to background diversity alone.

This is NOT length generalization: every input is exactly LENGTH characters. The
distribution shift is over WHERE X and Y may appear, controlled by SPLIT:

    SPLIT='none'   full distribution -- X,Y anywhere (with X < Y). Learnability baseline.

    SPLIT='half'   train/val : BOTH X and Y in the FIRST half   (length 6 -> 0..2)
                   test      : BOTH X and Y in the SECOND half  (length 6 -> 3..5)
                   Having learned "distance 1 vs larger" in the first half, does the model still
                   do it in unseen second-half positions?

SMALL HALVES AT LENGTH 6 -- EXPECTED, NOT A BUG
-----------------------------------------------
A contiguous region of size R admits R-1 distance-1 (T) configurations and C(R,2)-(R-1) gap
(F) configurations. At LENGTH=6 each half is only R=3 positions, so each pool has just
3 position configurations: 2 T (e.g. X@3Y@4, X@4Y@5) and 1 F (X@3Y@5). That is a known and
accepted property of the length-6 DEBUGGING anchor -- coverage grows in Stage 3 (length
8/10/12). The per-pool configuration counts are printed below so the coverage is on record.

SHORTCUT CONFOUND to watch (read this before believing a high score)
--------------------------------------------------------------------
With only 3 configurations in the training region, "distance 1 vs larger" is perfectly
correlated with a pure POSITION lookup: in 0..2 the only F is exactly (X@0, Y@2), so a model
can score 100% in-distribution by memorizing "F iff X@0 and Y@2" without ever comparing X
and Y. Two things guard against reading that as success:
  - the held-out pool is class-balanced 50/50, so a position-lookup model that answers T
    everywhere in the unseen half scores exactly 50% (chance), not 67%;
  - the per-position heatmap (evaluate.py -> plot.py) separates the two: accuracy tracking
    the GAP (constant along the diagonal) = the model reads relative distance; accuracy
    tracking WHERE Y sits = a position shortcut.
For T5 the shortcut risk is low in principle -- its bias is relative-distance based, and
the T case is distance 1, which its fine-grained near buckets capture exactly -- but that is an
expectation, so verify it on the heatmap rather than assuming it.

Config is overridable from the command line (poor-man's configurator, same pattern as
train.py / evaluate.py):
    ../venv/bin/python data/background/prepare.py --b=1
    ../venv/bin/python data/background/prepare.py --b=10 --length=12 --split=half

Outputs (in this folder):
    train.bin / val.bin  : flat token streams, one example per line (uint16); val uses the
                           SAME position rule as train (in-distribution monitoring)
    test.txt             : the held-out test examples, one per line, for evaluate.py
    meta.pkl             : vocab (stoi/itos/vocab_size) + split + length + b, so train.py and
                           evaluate.py never hardcode any of them
"""
import os
import pickle
import random
import sys
from ast import literal_eval
from itertools import combinations

import numpy as np

# ----------------------------- config (overridable) -----------------------------
SEED = 1337
LENGTH = 6             # fixed input length (slots before ':'); 6 = the debugging anchor
B = 1                  # BACKGROUND DIVERSITY: number of distinct background token types (1..10)

# Which region X and Y live in (the held-out axis is symbol POSITION, not length):
#   'none' : no held-out positions -> full distribution (learnability baseline)
#   'half' : train on first half, test on second half
SPLIT = 'half'

N_TRAIN = 20_000       # training examples
N_VAL = 2_000          # in-distribution val (monitoring during training)
N_TEST = 2_000         # held-out test examples (class-balanced, configuration-covered)
# --------------------------------------------------------------------------------

# poor-man's configurator: --key=value overrides a config key above. Lowercase aliases are
# accepted (--b, --length, --split) since that is how the run plan spells them.
_ALIASES = {'b': 'B', 'length': 'LENGTH', 'split': 'SPLIT', 'seed': 'SEED',
            'n_train': 'N_TRAIN', 'n_val': 'N_VAL', 'n_test': 'N_TEST'}
for arg in sys.argv[1:]:
    assert arg.startswith('--') and '=' in arg, f"expected --key=value, got {arg!r}"
    key, val = arg[2:].split('=', 1)
    key = _ALIASES.get(key, key)
    assert key in globals(), f"unknown config key: {key}"
    try:
        val = literal_eval(val)
    except (SyntaxError, ValueError):
        pass
    assert type(val) == type(globals()[key]), \
        f"type mismatch for {key}: {type(val)} vs {type(globals()[key])}"
    globals()[key] = val

DIGITS = '0123456789'
assert 1 <= B <= len(DIGITS), f"b must be in 1..{len(DIGITS)}, got {B}"
assert LENGTH >= 4, f"length must be >= 4 to leave room for a gap configuration, got {LENGTH}"
assert LENGTH % 2 == 0, f"length must be even so the halves are equal, got {LENGTH}"

BACKGROUND = DIGITS[:B]      # the size-b background token set: b=1 -> '0', b=3 -> '012'
HALF = LENGTH // 2           # first half = 0..HALF-1, second half = HALF..LENGTH-1

random.seed(SEED)
np.random.seed(SEED)


def label_for(x, y):
    """The fixed label rule: T iff the X-Y distance is exactly 1, else F."""
    return 'T' if y == x + 1 else 'F'


def allowed_positions(role):
    """Positions where X and Y may be placed for this pool. role is 'train' (also used for
    val) or 'test'. Returns a contiguous list of positions."""
    allpos = list(range(LENGTH))
    if SPLIT == 'none':
        return allpos
    if SPLIT == 'half':
        return allpos[HALF:] if role == 'test' else allpos[:HALF]
    raise ValueError(f"unknown SPLIT: {SPLIT!r}")


def configs_for(allowed, label):
    """Every (x, y) placement in `allowed` with x < y whose label matches `label`.
    X is always to the LEFT of Y, so we only ever take sorted pairs."""
    return [(x, y) for x, y in combinations(sorted(allowed), 2) if label_for(x, y) == label]


def make_example(x, y, label):
    """Build one '<LENGTH chars>:<T|F>' example with X at x, Y at y, background elsewhere."""
    chars = [random.choice(BACKGROUND) for _ in range(LENGTH)]
    chars[x] = 'X'
    chars[y] = 'Y'
    return ''.join(chars) + ':' + label


def make_balanced_pool(n, role):
    """n examples, class-balanced 50/50 T/F, configurations drawn uniformly within each
    class. Only the background tokens are re-randomized per example."""
    allowed = allowed_positions(role)
    by_label = {lab: configs_for(allowed, lab) for lab in ('T', 'F')}
    for lab, cfgs in by_label.items():
        assert cfgs, f"no {lab} configurations available in positions {allowed} -- " \
                     f"length {LENGTH} is too small for split {SPLIT!r}"
    examples = []
    for i in range(n):
        label = 'T' if i % 2 == 0 else 'F'
        x, y = random.choice(by_label[label])
        examples.append(make_example(x, y, label))
    random.shuffle(examples)
    return examples


def _split_count(total, items):
    """Distribute `total` as evenly as possible across `items`, summing to EXACTLY `total`
    (the remainder is spread one-per-item over the first few)."""
    k = len(items)
    base, rem = divmod(total, k)
    return {it: base + (1 if i < rem else 0) for i, it in enumerate(items)}


def make_config_balanced_test(n):
    """Held-out test pool: EXACTLY 50/50 T/F, and within each class spread as evenly as
    possible over that class's position configurations.

    Class balance is prioritized over equal-count-per-configuration when they conflict, so
    chance stays exactly 50% and a model that collapses to one label scores 50%. That is
    what makes the position-shortcut model (which answers T everywhere in the unseen half)
    land at chance instead of at an impressive-looking 67%."""
    allowed = allowed_positions('test')
    per_config, examples = {}, []
    for label in ('T', 'F'):
        cfgs = configs_for(allowed, label)
        counts = _split_count(n // 2, cfgs)          # sums to exactly n//2 for this class
        for cfg in cfgs:
            for _ in range(counts[cfg]):
                examples.append(make_example(cfg[0], cfg[1], label))
            per_config[cfg] = (counts[cfg], label)
    random.shuffle(examples)
    return examples, per_config


train_examples = make_balanced_pool(N_TRAIN, 'train')
val_examples = make_balanced_pool(N_VAL, 'train')     # in-distribution monitoring (train's rule)
test_examples, test_per_config = make_config_balanced_test(N_TEST)

# vocab is READ FROM THE BUILT DATA, never hardcoded: it grows with b (b=1 -> 7 tokens,
# b=10 -> 16), and train.py/evaluate.py pick it up from meta.pkl.
vocab_chars = sorted(set(''.join(train_examples + val_examples + test_examples)) | {'\n'})
vocab_size = len(vocab_chars)
stoi = {c: i for i, c in enumerate(vocab_chars)}
itos = {i: c for i, c in enumerate(vocab_chars)}


def encode(s):
    return [stoi[c] for c in s]


def write_bin(examples, path):
    """Concatenate examples into one flat stream (each ends with '\\n') and save token ids."""
    stream = ''.join(ex + '\n' for ex in examples)
    ids = np.array(encode(stream), dtype=np.uint16)
    ids.tofile(path)
    return len(ids)


here = os.path.dirname(os.path.abspath(__file__))
n_train_tok = write_bin(train_examples, os.path.join(here, 'train.bin'))
n_val_tok = write_bin(val_examples, os.path.join(here, 'val.bin'))

with open(os.path.join(here, 'test.txt'), 'w') as f:
    f.write('\n'.join(test_examples) + '\n')

# split_detail: a short human-readable tag identifying the held-out rule, logged to
# results.csv by evaluate.py so each row records which split produced it.
SPLIT_DETAIL = {'none': 'full', 'half': 'first->second'}[SPLIT]

with open(os.path.join(here, 'meta.pkl'), 'wb') as f:
    pickle.dump({'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos,
                 'split': SPLIT, 'split_detail': SPLIT_DETAIL,
                 'length': LENGTH, 'b': B, 'background': BACKGROUND}, f)


# ----------------------------- checks + report -----------------------------
def xy_positions(body):
    """(x_pos, y_pos) of the single X and single Y in a body string."""
    xs = [i for i, c in enumerate(body) if c == 'X']
    ys = [i for i, c in enumerate(body) if c == 'Y']
    assert len(xs) == 1 and len(ys) == 1, f"need exactly one X and one Y: {body!r}"
    return xs[0], ys[0]


def class_balance(examples):
    t = sum(1 for e in examples if e.endswith('T'))
    return t, len(examples) - t


def check_examples(examples):
    """Assert, for EVERY example: correct length; exactly one X and one Y; X strictly left of
    Y; the label is T iff Y is immediately after X; and background slots use only the size-b
    token set. Position determines the answer, so a placement/label/background bug has to be
    caught here, before training."""
    for e in examples:
        body, label = e.split(':')
        assert len(body) == LENGTH, f"body length {len(body)} != {LENGTH}: {e!r}"
        x, y = xy_positions(body)
        assert x < y, f"X must be left of Y (never Y...X): X@{x} Y@{y}  {e!r}"
        assert label_for(x, y) == label, \
            f"label/distance mismatch: X@{x} Y@{y} gap={y - x} label={label}  {e!r}"
        for i, c in enumerate(body):
            if i not in (x, y):
                assert c in BACKGROUND, \
                    f"background char {c!r} at {i} outside the b={B} set {BACKGROUND!r}: {e!r}"


def check_split(examples, role):
    """Assert the SPLIT position rule actually holds for this pool."""
    if SPLIT != 'half':
        return                                        # 'none' -> no positional constraint
    for e in examples:
        x, y = xy_positions(e.split(':')[0])
        if role == 'train':
            assert x < HALF and y < HALF, f"train/val X,Y outside first half: {e!r}"
        else:
            assert x >= HALF and y >= HALF, f"test X,Y outside second half: {e!r}"


def rule_str(role):
    if SPLIT == 'none':
        return f"X,Y anywhere in 0..{LENGTH - 1} (X left of Y)"
    region = f"{HALF}..{LENGTH - 1} (second half)" if role == 'test' else f"0..{HALF - 1} (first half)"
    return f"both X,Y in {region}"


print("=== distance-1 task data prepared ===")
print(f"SPLIT={SPLIT}  LENGTH={LENGTH}  b={B}  SEED={SEED}")
print(f"background set: {list(BACKGROUND)}  ({B} token type{'s' if B > 1 else ''})")
print(f"vocab_size={vocab_size}  chars={[repr(c) for c in vocab_chars]}")
print("label rule    : T iff the X-Y distance is exactly 1, else F")
print(f"train/val rule: {rule_str('train')}")
print(f"test     rule : {rule_str('test')}")
print()

for name, ex, role in [('train', train_examples, 'train'),
                       ('val', val_examples, 'train'),
                       ('test', test_examples, 'test')]:
    check_examples(ex)                                # raises on any label/placement/bg bug
    check_split(ex, role)                             # raises if the split rule is violated
    t, fa = class_balance(ex)
    cfgs = sorted({xy_positions(e.split(':')[0]) for e in ex})
    n_t_cfg = sum(1 for c in cfgs if label_for(*c) == 'T')
    print(f"{name:5s}: {len(ex):>6d} examples | T(dist=1)={t} F(gap)={fa} ({t / len(ex):.1%} T)"
          f" | {len(cfgs)} configs ({n_t_cfg} T, {len(cfgs) - n_t_cfg} F)"
          f" | {len(set(ex))} distinct examples")
print("X-before-Y, label-correctness and background-set assertions passed for all pools")
print("split-rule assertion passed for all pools")
print()

# Per-configuration coverage of the held-out pool. At LENGTH=6 this is only 3 rows (2 T,
# 1 F) -- expected for the debugging anchor; Stage 3's longer inputs grow it.
print(f"held-out test pool, per-configuration coverage ({rule_str('test')}):")
print("  X@ , Y@  gap : count  (label)")
for (x, y), (cnt, label) in sorted(test_per_config.items()):
    kind = 'T dist=1'   if label == 'T' else 'F gap     '
    print(f"  X@{x:<2} Y@{y:<2} gap={y - x:<2}: {cnt:>5}  ({kind})")
t, fa = class_balance(test_examples)
print(f"  class totals: T={t}  F={fa}  -> exact 50/50 = {t == fa}")
print(f"  ({len(test_per_config)} distinct configurations; at length 6 each half has only 3 "
      f"-- expected, not a blocker)")
print()

print(f"train.bin: {n_train_tok:,} tokens | val.bin: {n_val_tok:,} tokens | "
      f"test.txt: {len(test_examples)} lines")


def show(examples, k=6):
    for e in examples[:k]:
        body, label = e.split(':')
        x, y = xy_positions(body)
        kind = 'dist=1  ' if y == x + 1 else 'gap     '
        print(f"  X@{x:>2} Y@{y:>2}  gap={y - x:>2} ({kind}) -> {label}   {e}")


print("\nFirst 6 train examples (X/Y positions + gap marked):")
show(train_examples)
print("First 6 test examples (held-out):")
show(test_examples)
