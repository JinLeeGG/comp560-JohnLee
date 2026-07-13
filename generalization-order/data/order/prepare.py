"""
Relative-order task data generator.

Task: a fixed-length string of digits that contains the symbol X EXACTLY ONCE and
the symbol Y EXACTLY ONCE (the other 18 characters are random digits 0-9). The
model reads the 20-char string, sees ':', and must output a SINGLE token:
    LABEL MAPPING (fixed -- never flip this):
        T  if  index(X) < index(Y)   (X comes before Y)
        F  otherwise                 (Y comes before X)

    4829X017364Y19285746:T   <- X at 4, Y at 11  -> X before Y -> T
    4829Y017364X19285746:F   <- Y at 4, X at 11  -> Y before X -> F

This is NOT length generalization: every input is exactly LENGTH=20 characters. The
distribution shift is over WHERE the special symbols may appear, controlled by SPLIT:

    SPLIT='none'        full distribution -- X and Y in any distinct position pair.
                        (Baseline: both 'none' and 'learned' PE solved this at 100%,
                        because a causal decoder gets order from the causal mask.)

    SPLIT='single_pos'  hold out ONE position P (default 12), generalizing detection's
                        position-12 hold-out to two symbols:
                          train/val : NEITHER X nor Y is ever at P
                          test      : exactly one of X/Y sits at P (the other elsewhere)
                        All four sub-cases occur in test and are needed for balance:
                          X@P, Y before -> F   X@P, Y after  -> T
                          Y@P, X before -> T   Y@P, X after  -> F

    SPLIT='half'        train/val : BOTH X and Y in the first half (0..9)
                        test      : BOTH X and Y in the second half (10..19)
                        Harder: special symbols are never seen in the 2nd half in train.

The held-out question: on the full distribution the causal mask's implicit ordering
is learned on the SEEN positions. Does it still hold at positions never seen? If
'none' drops toward chance on test while 'learned' holds, that is the phenomenon.

Outputs (in this folder):
    train.bin / val.bin  : flat token streams, one example per line (uint16); val uses
                           the SAME position rule as train (in-distribution monitoring)
    test.txt             : the held-out test examples, one per line, for evaluate.py
    meta.pkl             : vocab (stoi/itos/vocab_size) for encode/decode
"""
import os
import pickle
import random
import numpy as np

# ----------------------------- config -----------------------------
# DATA split seed. Default 1337 (unchanged behavior). Set env DATA_SEED to
# regenerate the split per model seed, so multi-seed variance includes
# data-sampling variance, not just model-init/batch-order noise.
SEED = int(os.environ.get('DATA_SEED', '1337'))
LENGTH = 20            # fixed input length (number of char slots before ':')

# Which positions are HELD OUT (used only in the test set, never in training).
#   'none'        : no held-out positions -> full distribution (baseline)
#   'single_pos'  : hold out one position HELDOUT_POS (option B)
#   'half'        : train on first half, test on second half (option C)
SPLIT = 'half'
HELDOUT_POS = 12       # position P held out, used only when SPLIT == 'single_pos'

N_TRAIN = 50_000       # number of training examples
N_VAL = 5_000          # in-distribution val (monitoring during training)
N_TEST = 2_000         # held-out test examples
# ------------------------------------------------------------------

HALF = LENGTH // 2     # first half = 0..HALF-1, second half = HALF..LENGTH-1

random.seed(SEED)
np.random.seed(SEED)

# vocab: digits 0-9, plus X,Y (symbols), T,F (labels), ':' (delimiter), '\n' (separator)
DIGITS = '0123456789'
vocab_chars = sorted(set(DIGITS + 'X' + 'Y' + 'T' + 'F' + ':' + '\n'))
vocab_size = len(vocab_chars)          # 10 digits + X,Y,T,F + ':' + '\n' = 16
stoi = {c: i for i, c in enumerate(vocab_chars)}
itos = {i: c for i, c in enumerate(vocab_chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return ''.join(itos[i] for i in ids)


def allowed_positions(role):
    """Positions where a special symbol (X or Y) may be placed for this pool.
    role is 'train' (also used for val) or 'test'. Returns a list of positions, or
    None for the single_pos test pool (special-cased: exactly one symbol must be at P)."""
    allpos = list(range(LENGTH))
    if SPLIT == 'none':
        return allpos
    if SPLIT == 'single_pos':
        if role == 'test':
            return None                                  # special-cased below
        return [p for p in allpos if p != HELDOUT_POS]   # train/val: never at P
    if SPLIT == 'half':
        return allpos[HALF:] if role == 'test' else allpos[:HALF]
    raise ValueError(f"unknown SPLIT: {SPLIT!r}")


def place_from_allowed(allowed, label):
    """Pick two distinct positions from `allowed` and assign X/Y to realize `label`.
    Returns (pos_x, pos_y) with  (pos_x < pos_y) iff label == 'T'."""
    lo, hi = sorted(random.sample(allowed, 2))
    return (lo, hi) if label == 'T' else (hi, lo)        # T: X before Y; F: Y before X


def place_single_pos_test(label):
    """single_pos test: exactly one of X/Y sits at HELDOUT_POS, the other elsewhere.
    Returns (pos_x, pos_y) realizing `label`. Randomizes which symbol sits at P so all
    four sub-cases appear; falls back to the other symbol if P is at a boundary."""
    P = HELDOUT_POS
    before = list(range(0, P))            # positions < P
    after = list(range(P + 1, LENGTH))    # positions > P
    for at_P in random.sample(['X', 'Y'], 2):            # try both, in random order
        if at_P == 'X':                                  # X@P; T needs Y>P, F needs Y<P
            other = after if label == 'T' else before
            if other:
                return P, random.choice(other)           # (pos_x=P, pos_y)
        else:                                            # Y@P; T needs X<P, F needs X>P
            other = before if label == 'T' else after
            if other:
                return random.choice(other), P           # (pos_x, pos_y=P)
    raise ValueError(f"cannot place single_pos test: P={P}, label={label} (boundary?)")


def make_example(role, label):
    """Build one example string '<LENGTH chars>:<T|F>' obeying the SPLIT rule for `role`."""
    if SPLIT == 'single_pos' and role == 'test':
        pos_x, pos_y = place_single_pos_test(label)
    else:
        pos_x, pos_y = place_from_allowed(allowed_positions(role), label)
    chars = [random.choice(DIGITS) for _ in range(LENGTH)]
    chars[pos_x] = 'X'
    chars[pos_y] = 'Y'
    return ''.join(chars) + ':' + label


def make_pool(n, role):
    """Build n examples, class-balanced 50/50 between T (X before Y) and F (Y before X)."""
    examples = [make_example(role, 'T' if i % 2 == 0 else 'F') for i in range(n)]
    random.shuffle(examples)
    return examples


train_examples = make_pool(N_TRAIN, 'train')
val_examples = make_pool(N_VAL, 'train')       # in-distribution monitoring (same rule as train)
test_examples = make_pool(N_TEST, 'test')      # held-out generalization test


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
SPLIT_DETAIL = {'none': 'full',
                'single_pos': f'P={HELDOUT_POS}',
                'half': 'first->second'}[SPLIT]

with open(os.path.join(here, 'meta.pkl'), 'wb') as f:
    pickle.dump({'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos,
                 'split': SPLIT, 'split_detail': SPLIT_DETAIL}, f)


# ----------------------------- checks + report -----------------------------
def class_balance(examples):
    t = sum(1 for e in examples if e.endswith('T'))
    return t, len(examples) - t


def check_labels(examples):
    """Assert (index(X) < index(Y)) iff label is 'T' for EVERY example. Position now
    determines the answer, so a placement/label bug must be caught before training."""
    for e in examples:
        body, label = e.split(':')
        assert len(body) == LENGTH, f"body length {len(body)} != {LENGTH}: {e!r}"
        assert body.count('X') == 1 and body.count('Y') == 1, f"need one X and one Y: {e!r}"
        x_before_y = body.index('X') < body.index('Y')
        assert x_before_y == (label == 'T'), \
            f"label/position mismatch: X@{body.index('X')} Y@{body.index('Y')} label={label}  {e!r}"


def check_split(examples, role):
    """Assert the SPLIT position rule actually holds for this pool."""
    for e in examples:
        body = e.split(':')[0]
        xp, yp = body.index('X'), body.index('Y')
        if SPLIT == 'single_pos':
            if role == 'train':
                assert xp != HELDOUT_POS and yp != HELDOUT_POS, \
                    f"train/val has a symbol at held-out P={HELDOUT_POS}: {e!r}"
            else:  # test: exactly one symbol at P
                assert (xp == HELDOUT_POS) ^ (yp == HELDOUT_POS), \
                    f"test must have exactly one symbol at P={HELDOUT_POS}: {e!r}"
        elif SPLIT == 'half':
            if role == 'train':
                assert xp < HALF and yp < HALF, f"train/val symbol outside first half: {e!r}"
            else:
                assert xp >= HALF and yp >= HALF, f"test symbol outside second half: {e!r}"
        # SPLIT == 'none' -> no positional constraint


def rule_str(role):
    if SPLIT == 'none':
        return "X,Y in any positions 0..19"
    if SPLIT == 'single_pos':
        if role == 'train':
            return f"neither X nor Y at P={HELDOUT_POS}"
        return f"exactly one of X/Y at P={HELDOUT_POS}"
    if SPLIT == 'half':
        return "both X,Y in 0..9 (first half)" if role == 'train' else "both X,Y in 10..19 (second half)"


print("=== Relative-order data prepared ===")
print(f"SPLIT={SPLIT}  LENGTH={LENGTH}  SEED={SEED}" +
      (f"  HELDOUT_POS={HELDOUT_POS}" if SPLIT == 'single_pos' else ''))
vocab_display = [repr(c) for c in vocab_chars]
print(f"vocab_size={vocab_size}  chars={vocab_display}")
print(f"train/val rule: {rule_str('train')}")
print(f"test     rule : {rule_str('test')}")
if SPLIT == 'half':
    print(f"note: each half has 10 positions -> 10*9 = 90 ordered (X,Y) pairs available")
print()
for name, ex, role in [('train', train_examples, 'train'),
                       ('val', val_examples, 'train'),
                       ('test', test_examples, 'test')]:
    check_labels(ex)                                  # raises if any label is wrong
    check_split(ex, role)                             # raises if the split rule is violated
    t, fa = class_balance(ex)
    print(f"{name:5s}: {len(ex):>6d} examples  | T={t} F={fa}  ({t / len(ex):.1%} T)")
print("label-correctness assertion passed for all pools  (index(X)<index(Y) iff T)")
print("split-rule assertion passed for all pools")
print()
print(f"train.bin: {n_train_tok:,} tokens | val.bin: {n_val_tok:,} tokens | test.txt: {len(test_examples)} lines")


def show(examples, k=6):
    for e in examples[:k]:
        body, label = e.split(':')
        xp, yp = body.index('X'), body.index('Y')
        print(f"  X@{xp:>2} Y@{yp:>2} -> {label}   {e}")


print("\nFirst 6 train examples (X/Y positions marked):")
show(train_examples)
print("First 6 test examples (held-out):")
show(test_examples)
