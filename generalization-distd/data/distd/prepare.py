"""
Distance-threshold task data generator (distance task #2: dist >= D).

Task: a fixed-length string of digits that contains the symbol X EXACTLY TWICE (two
identical targets; the other 18 characters are random digits 0-9). The model reads the
20-char string, sees ':', and must output a SINGLE token:

    DISTANCE  = |pos2 - pos1|     (absolute difference of the two X positions)
    THRESHOLD = D = 5             (fixed -- the one constant that defines this task)
    LABEL MAPPING (fixed -- never flip this):
        T  if  DISTANCE >= D   (the two X's are FAR apart,  >= 5)
        F  if  DISTANCE <  D   (the two X's are NEAR,        <= 4)

    482X50178X4019285746:T   <- X at 3, X at 9  -> distance 6 >= 5  -> T (far)
    482X5X178640192857 46:F  <- X at 3, X at 5  -> distance 2 <  5  -> F (near)

Why this task (contrast with even/odd, distance task #1): even/odd asks for the EXACT
parity of the distance ("off by one flips the label"), which a micro (<1M) model could not
learn at all -- every PE/seed stuck at 50% in-distribution val (a capacity wall). dist>=D
is COARSER: it needs only an approximate, monotone sense of "far vs near", not the exact
value, so it may fit under the same micro budget where parity did not. Honest expectation:
this is close to a coin flip -- if distance-READING itself (not parity specifically) is the
micro bottleneck, dist>=D hits the same wall. Either outcome is informative.

Why two identical X's (not X and Y): distance is order-invariant, so a second distinct
symbol would only add an irrelevant order cue. Two X's isolate distance cleanly.

This is NOT length generalization: every input is exactly LENGTH=20 characters. The
distribution shift is over WHERE the two X's may appear, controlled by SPLIT:

    SPLIT='none'   full distribution -- both X's in any distinct position pair. Baseline.
                   (In the full distribution far distances dominate, so T would naturally
                   outnumber F; we FORCE 50/50 by sampling per label.)

    SPLIT='half'   train/val : BOTH X's in the FIRST half (0..9)
                   test      : BOTH X's in the SECOND half (10..19)
                   Having learned dist>=D in the first half, does the model still do it in
                   unseen second-half positions? The expected story: NoPE ('none') reads
                   ORDER from the causal mask but cannot COUNT distance, so it should fail
                   on the held-out half; distance-aware (rope/t5) and absolute methods
                   (learned/sinusoidal, which can subtract positions) may hold up. BUT
                   dist>=D is coarse enough that NoPE might approximate "are the two X's
                   roughly far apart" -- if so, that is a finding, not a failure.

CONFOUND to watch (more exposed here than in even/odd): on the 'half' split, "distance >= 5"
and "the two X's sit at opposite ENDS of 10..19" are highly correlated. A model could score
well by detecting a POSITION pattern (one X left, one X right) instead of measuring distance.
The per-position heatmap distinguishes them: genuine distance -> BANDS parallel to the
diagonal; position shortcut -> BLOCKS. (Reported in evaluate.py / plot.py output.)

Class / distance balance:
  - 50/50 T/F in EVERY pool (enforced, not assumed).
  - Held-out test prioritizes exact 50/50 class balance (so chance stays interpretable) and
    then spreads examples as evenly as possible across the distances inside each class side.
    The resulting per-distance counts are printed so any skew is visible. For D=5 in 10..19:
    distances 1,2,3,4 -> F (250 each), distances 5,6,7,8,9 -> T (200 each), giving exactly
    1000/1000. This is NOT equal-count across all nine distances; exact class balance wins.

Outputs (in this folder):
    train.bin / val.bin  : flat token streams, one example per line (uint16); val uses
                           the SAME position rule as train (in-distribution monitoring)
    test.txt             : the held-out test examples, one per line, for evaluate.py
    meta.pkl             : vocab (stoi/itos/vocab_size) + split + threshold D for encode/decode
"""
import os
import pickle
import random
import numpy as np

# ----------------------------- config -----------------------------
SEED = 1337
LENGTH = 20            # fixed input length (number of char slots before ':')
D = 5                  # THRESHOLD: T iff distance >= D, else F. The one constant of this task.

# Which region the two X's live in (the held-out axis is symbol POSITION, not length):
#   'none' : no held-out positions -> full distribution (baseline)
#   'half' : train on first half (0..9), test on second half (10..19)
SPLIT = 'half'

N_TRAIN = 50_000       # number of training examples
N_VAL = 5_000          # in-distribution val (monitoring during training)
N_TEST = 2_000         # held-out test examples (class-balanced, distance-covered for half)
# ------------------------------------------------------------------

HALF = LENGTH // 2     # first half = 0..HALF-1, second half = HALF..LENGTH-1

random.seed(SEED)
np.random.seed(SEED)

# vocab: digits 0-9, plus X (the doubled symbol), T,F (labels), ':' (delimiter), '\n'.
# No 'Y' -- the two targets are identical, so order carries no information.
DIGITS = '0123456789'
vocab_chars = sorted(set(DIGITS + 'X' + 'T' + 'F' + ':' + '\n'))
vocab_size = len(vocab_chars)          # 10 digits + X,T,F + ':' + '\n' = 15
stoi = {c: i for i, c in enumerate(vocab_chars)}
itos = {i: c for i, c in enumerate(vocab_chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return ''.join(itos[i] for i in ids)


def allowed_positions(role):
    """Positions where an X may be placed for this pool. role is 'train' (also used for
    val) or 'test'. Returns a contiguous list of positions."""
    allpos = list(range(LENGTH))
    if SPLIT == 'none':
        return allpos
    if SPLIT == 'half':
        return allpos[HALF:] if role == 'test' else allpos[:HALF]
    raise ValueError(f"unknown SPLIT: {SPLIT!r}")


def label_for(lo, hi):
    """The fixed label rule: T iff the distance |hi-lo| is >= D (far), else F (near)."""
    return 'T' if (hi - lo) >= D else 'F'


def positions_for_label(allowed, label):
    """Pick two distinct positions from `allowed` whose distance threshold matches `label`
    (T = dist>=D far, F = dist<D near). Rejection sampling. Returns sorted (lo, hi)."""
    while True:
        lo, hi = sorted(random.sample(allowed, 2))
        if label_for(lo, hi) == label:
            return lo, hi


def positions_for_distance(allowed, dist):
    """Pick two positions in `allowed` (a contiguous range) exactly `dist` apart.
    Returns sorted (lo, hi)."""
    allowed_set = set(allowed)
    candidates = [(p, p + dist) for p in allowed if (p + dist) in allowed_set]
    return random.choice(candidates)


def make_example(lo, hi, label):
    """Build one '<LENGTH chars>:<T|F>' example with X at lo and X at hi."""
    chars = [random.choice(DIGITS) for _ in range(LENGTH)]
    chars[lo] = 'X'
    chars[hi] = 'X'
    return ''.join(chars) + ':' + label


def make_balanced_pool(n, role):
    """Build n examples, class-balanced 50/50 between T (dist>=D) and F (dist<D). Distances
    within each class are whatever rejection sampling yields -- training WANTS varied
    distances; only the held-out test pool needs explicit per-distance balance."""
    allowed = allowed_positions(role)
    examples = []
    for i in range(n):
        label = 'T' if i % 2 == 0 else 'F'
        lo, hi = positions_for_label(allowed, label)
        examples.append(make_example(lo, hi, label))
    random.shuffle(examples)
    return examples


def _split_count(total, distances):
    """Distribute `total` examples as evenly as possible across `distances`, so the sum is
    EXACTLY `total` (remainder spread one-per-distance over the first few). Returns a
    {distance: count} dict. This is what lets us force exact class balance while staying as
    distance-covered as the arithmetic allows."""
    k = len(distances)
    base, rem = divmod(total, k)
    return {d: base + (1 if i < rem else 0) for i, d in enumerate(distances)}


def make_distance_balanced_test(n):
    """Held-out test pool with both X's in the second half (SPLIT='half'), balanced across
    distances within each class and EXACTLY 50/50 T/F (class balance prioritized over global
    equal-count-per-distance balance when they conflict).

    A contiguous region of size R has distances 1..R-1. Distances < D -> F (near), distances
    >= D -> T (far). Each class gets exactly n//2 examples, spread as evenly as possible over
    its distances. With R=10, D=5, n=2000: F-distances {1,2,3,4} -> 250 each (F=1000);
    T-distances {5,6,7,8,9} -> 200 each (T=1000)."""
    allowed = allowed_positions('test')
    R = len(allowed)
    dists = list(range(1, R))                          # 1..R-1
    near = [d for d in dists if d < D]                 # -> F
    far = [d for d in dists if d >= D]                 # -> T
    half_n = n // 2
    near_counts = _split_count(half_n, near)           # sums to exactly half_n  (F class)
    far_counts = _split_count(half_n, far)             # sums to exactly half_n  (T class)
    examples, per_distance = [], {}
    for d in dists:
        cnt = far_counts[d] if d >= D else near_counts[d]
        label = 'T' if d >= D else 'F'
        for _ in range(cnt):
            lo, hi = positions_for_distance(allowed, d)
            examples.append(make_example(lo, hi, label))
        per_distance[d] = cnt
    random.shuffle(examples)
    return examples, per_distance


train_examples = make_balanced_pool(N_TRAIN, 'train')
val_examples = make_balanced_pool(N_VAL, 'train')     # in-distribution monitoring (same rule as train)
if SPLIT == 'half':
    test_examples, test_per_distance = make_distance_balanced_test(N_TEST)
else:                                                 # SPLIT == 'none': full-dist, class-balanced
    test_examples = make_balanced_pool(N_TEST, 'test')
    test_per_distance = None


def write_bin(examples, path):
    """Concatenate examples into one flat stream (each ends with '\\n') and save token ids."""
    stream = ''.join(ex + '\n' for ex in examples)
    ids = np.array(encode(stream), dtype=np.uint16)
    ids.tofile(path)
    return len(ids)


here = os.path.dirname(__file__)
n_train_tok = write_bin(train_examples, os.path.join(here, 'train.bin'))
n_val_tok = write_bin(val_examples, os.path.join(here, 'val.bin'))

# test set: keep human-readable, one example per line, for evaluate.py
with open(os.path.join(here, 'test.txt'), 'w') as f:
    f.write('\n'.join(test_examples) + '\n')

# split_detail: a short human-readable tag identifying the held-out rule, logged to
# results.csv by evaluate.py so each row records which split produced it.
SPLIT_DETAIL = {'none': 'full', 'half': 'first->second'}[SPLIT]

with open(os.path.join(here, 'meta.pkl'), 'wb') as f:
    pickle.dump({'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos,
                 'split': SPLIT, 'split_detail': SPLIT_DETAIL, 'threshold': D}, f)


# ----------------------------- checks + report -----------------------------
def x_positions(body):
    return [i for i, c in enumerate(body) if c == 'X']


def class_balance(examples):
    t = sum(1 for e in examples if e.endswith('T'))
    return t, len(examples) - t


def check_labels(examples):
    """Assert |pos2 - pos1| >= D iff label is 'T' for EVERY example, and that exactly two
    X's are present. Position determines the answer, so a placement/label bug must be caught
    before training."""
    for e in examples:
        body, label = e.split(':')
        assert len(body) == LENGTH, f"body length {len(body)} != {LENGTH}: {e!r}"
        xs = x_positions(body)
        assert len(xs) == 2, f"need exactly two X's, found {len(xs)}: {e!r}"
        lo, hi = xs
        assert label_for(lo, hi) == label, \
            f"label/distance mismatch: X@{lo} X@{hi} dist={hi - lo} label={label}  {e!r}"


def check_split(examples, role):
    """Assert the SPLIT position rule actually holds for this pool."""
    for e in examples:
        body = e.split(':')[0]
        lo, hi = x_positions(body)
        if SPLIT == 'half':
            if role == 'train':
                assert lo < HALF and hi < HALF, f"train/val X outside first half: {e!r}"
            else:
                assert lo >= HALF and hi >= HALF, f"test X outside second half: {e!r}"
        # SPLIT == 'none' -> no positional constraint


def rule_str(role):
    if SPLIT == 'none':
        return "both X's in any positions 0..19"
    if SPLIT == 'half':
        return "both X's in 0..9 (first half)" if role == 'train' else "both X's in 10..19 (second half)"


print("=== dist>=D threshold data prepared ===")
print(f"SPLIT={SPLIT}  LENGTH={LENGTH}  D={D}  SEED={SEED}")
vocab_display = [repr(c) for c in vocab_chars]
print(f"vocab_size={vocab_size}  chars={vocab_display}")
print(f"label rule    : T iff |pos2-pos1| >= {D} (far), else F (near)")
print(f"train/val rule: {rule_str('train')}")
print(f"test     rule : {rule_str('test')}")
print()
for name, ex, role in [('train', train_examples, 'train'),
                       ('val', val_examples, 'train'),
                       ('test', test_examples, 'test')]:
    check_labels(ex)                                  # raises if any label is wrong
    check_split(ex, role)                             # raises if the split rule is violated
    t, fa = class_balance(ex)
    print(f"{name:5s}: {len(ex):>6d} examples  | T(far>={D})={t} F(near<{D})={fa}  ({t / len(ex):.1%} T)")
print(f"label-correctness assertion passed for all pools  (|pos2-pos1| >= {D} iff T)")
print("split-rule assertion passed for all pools")
print()

if test_per_distance is not None:
    print("held-out test pool, per-distance coverage (both X's in second half 10..19):")
    print("  dist : count  (label)")
    for d in sorted(test_per_distance):
        lab = f'T far ' if d >= D else 'F near'
        print(f"  {d:>4} : {test_per_distance[d]:>5}  ({lab})")
    t, fa = class_balance(test_examples)
    print(f"  (distances {min(test_per_distance)}..{max(test_per_distance)}; "
          f"each has >= {min(test_per_distance.values())} examples for the separation graph)")
    print(f"  class totals from these counts: T(far)={t}  F(near)={fa}  -> exact 50/50 = {t == fa}")
    print()

print(f"train.bin: {n_train_tok:,} tokens | val.bin: {n_val_tok:,} tokens | test.txt: {len(test_examples)} lines")


def show(examples, k=6):
    for e in examples[:k]:
        body, label = e.split(':')
        lo, hi = x_positions(body)
        far = 'far ' if (hi - lo) >= D else 'near'
        print(f"  X@{lo:>2} X@{hi:>2}  dist={hi - lo:>2} ({far}) -> {label}   {e}")


print("\nFirst 6 train examples (X positions + distance marked):")
show(train_examples)
print("First 6 test examples (held-out):")
show(test_examples)
