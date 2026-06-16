"""
Detection task data generator.

Task: a fixed-length string of digits, with the special symbol X present or not.
    48295017364X19285746:Y   <- X is present (at one allowed position)  -> label Y
    48295017364019285746:N   <- no X (all digits)                       -> label N
The model reads the 20-digit string, sees ':', and must output a SINGLE token: Y or N.

This is NOT length generalization: every input is exactly LENGTH=20 characters.
The experiment splits the data by WHERE the X is allowed to appear:
    - train / val  : X only in "allowed" positions
    - test         : X only in "held-out" positions (never seen in training)
The SPLIT knob below selects which positions are held out.

Outputs (in this folder):
    train.bin / val.bin  : flat token streams for nanoGPT's get_batch (uint16)
    test.txt             : the generalization-test examples, one per line, for evaluate.py
    meta.pkl             : vocab (stoi/itos/vocab_size) for encode/decode
"""
import os
import pickle
import random
import numpy as np

# ----------------------------- config -----------------------------
SEED = 1337
LENGTH = 20            # fixed input length (number of digit slots before ':')

# Which positions are HELD OUT (used only in the test set, never in training).
#   'none'     : no held-out positions -> train on the FULL distribution (Phase 0 baseline)
#   'single'   : hold out one position (HELDOUT_SINGLE)
#   'even_odd' : train on even positions, test on odd positions
#   'half'     : train on first half, test on second half
SPLIT = 'single'       # 'single' = hold out HELDOUT_SINGLE for the generalization test
HELDOUT_SINGLE = 12    # only used when SPLIT == 'single'

N_TRAIN = 50_000       # number of training examples
N_VAL = 5_000          # in-distribution val (monitoring during training)
N_TEST = 2_000         # generalization-test examples
# ------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)

# vocab: digits 0-9, plus X (symbol), Y/N (labels), ':' (delimiter), '\n' (separator)
DIGITS = '0123456789'
vocab_chars = sorted(set(DIGITS + 'X' + 'Y' + 'N' + ':' + '\n'))
vocab_size = len(vocab_chars)          # 10 digits + X,Y,N + ':' + '\n' = 15
stoi = {c: i for i, c in enumerate(vocab_chars)}
itos = {i: c for i, c in enumerate(vocab_chars)}

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return ''.join(itos[i] for i in ids)


def get_positions(split):
    """Return (train_positions, test_positions): the lists of positions where X may appear."""
    allpos = list(range(LENGTH))
    if split == 'none':
        return allpos, allpos                      # test is in-distribution (baseline check)
    if split == 'single':
        train = [p for p in allpos if p != HELDOUT_SINGLE]
        return train, [HELDOUT_SINGLE]
    if split == 'even_odd':
        return [p for p in allpos if p % 2 == 0], [p for p in allpos if p % 2 == 1]
    if split == 'half':
        h = LENGTH // 2
        return allpos[:h], allpos[h:]
    raise ValueError(f"unknown SPLIT: {split}")


def make_example(x_positions, has_x):
    """Build one example string '<LENGTH digits>:<Y|N>'.
    If has_x, place exactly one X at a random position drawn from x_positions."""
    chars = [random.choice(DIGITS) for _ in range(LENGTH)]
    if has_x:
        p = random.choice(x_positions)
        chars[p] = 'X'
        label = 'Y'
    else:
        label = 'N'
    return ''.join(chars) + ':' + label


def make_pool(n, x_positions):
    """Build n examples, class-balanced 50/50 between X-present (Y) and X-absent (N)."""
    examples = [make_example(x_positions, has_x=(i % 2 == 0)) for i in range(n)]
    random.shuffle(examples)
    return examples


train_positions, test_positions = get_positions(SPLIT)
train_examples = make_pool(N_TRAIN, train_positions)
val_examples = make_pool(N_VAL, train_positions)     # in-distribution monitoring
test_examples = make_pool(N_TEST, test_positions)    # generalization test


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

with open(os.path.join(here, 'meta.pkl'), 'wb') as f:
    pickle.dump({'vocab_size': vocab_size, 'stoi': stoi, 'itos': itos}, f)


# ----------------------------- report -----------------------------
def class_balance(examples):
    y = sum(1 for e in examples if e.endswith('Y'))
    return y, len(examples) - y

print("=== Detection data prepared ===")
print(f"SPLIT={SPLIT}  LENGTH={LENGTH}  SEED={SEED}")
vocab_display = [repr(c) for c in vocab_chars]
print(f"vocab_size={vocab_size}  chars={vocab_display}")
print(f"train X-positions ({len(train_positions)}): {train_positions}")
print(f"test  X-positions ({len(test_positions)}): {test_positions}")
print()
for name, ex in [('train', train_examples), ('val', val_examples), ('test', test_examples)]:
    y, n = class_balance(ex)
    print(f"{name:5s}: {len(ex):>6d} examples  | Y={y} N={n}")
print()
print(f"train.bin: {n_train_tok:,} tokens | val.bin: {n_val_tok:,} tokens | test.txt: {len(test_examples)} lines")
print("\nFirst 5 train examples:")
for e in train_examples[:5]:
    print("  ", e)
print("First 5 test examples:")
for e in test_examples[:5]:
    print("  ", e)
