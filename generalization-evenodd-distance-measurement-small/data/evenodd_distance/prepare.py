"""
Small X/Y even/odd distance measurement sanity dataset.

Task: fixed-length string with one X and one Y, padded by O. Output:
    T iff abs(index(X) - index(Y)) is even, else F.

This is the smallest version of the advisor's 6/30 suggestion:
    - LENGTH = 6
    - vocab = O, X, Y, T, F, :, newline
    - train/val: both special symbols in positions 0..2
    - test:      both special symbols in positions 3..5

There is no length generalization here. Every example has exactly six body tokens.
The only distribution shift is over the absolute positions occupied by X and Y.
"""
import os
import pickle
import random

import numpy as np

# ----------------------------- config -----------------------------
SEED = 1337
LENGTH = 6
TRAIN_POSITIONS = [0, 1, 2]
TEST_POSITIONS = [3, 4, 5]

N_TRAIN = 600
N_VAL = 120
N_TEST = 120
# ------------------------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)

VOCAB_CHARS = ['\n', ':', 'F', 'O', 'T', 'X', 'Y']
stoi = {c: i for i, c in enumerate(VOCAB_CHARS)}
itos = {i: c for i, c in enumerate(VOCAB_CHARS)}
vocab_size = len(VOCAB_CHARS)


def encode(s):
    return [stoi[c] for c in s]


def ordered_pairs(positions):
    """All ordered (X position, Y position) pairs from a position set."""
    return [(xp, yp) for xp in positions for yp in positions if xp != yp]


def label_for(xp, yp):
    return 'T' if abs(xp - yp) % 2 == 0 else 'F'


def make_example(xp, yp):
    chars = ['O'] * LENGTH
    chars[xp] = 'X'
    chars[yp] = 'Y'
    label = label_for(xp, yp)
    return ''.join(chars) + ':' + label


def make_pool(n, positions):
    """Build a class-balanced pool by repeating even-distance and odd-distance pairs."""
    pairs_by_label = {'T': [], 'F': []}
    for xp, yp in ordered_pairs(positions):
        pairs_by_label[label_for(xp, yp)].append((xp, yp))
    assert pairs_by_label['T'] and pairs_by_label['F'], \
        f"need both even and odd distances in {positions}"
    examples = []
    for i in range(n):
        label = 'T' if i % 2 == 0 else 'F'
        xp, yp = pairs_by_label[label][(i // 2) % len(pairs_by_label[label])]
        examples.append(make_example(xp, yp))
    random.shuffle(examples)
    return examples


train_examples = make_pool(N_TRAIN, TRAIN_POSITIONS)
val_examples = make_pool(N_VAL, TRAIN_POSITIONS)
test_examples = make_pool(N_TEST, TEST_POSITIONS)


def write_bin(examples, path):
    stream = ''.join(ex + '\n' for ex in examples)
    ids = np.array(encode(stream), dtype=np.uint16)
    ids.tofile(path)
    return len(ids)


here = os.path.dirname(__file__)
n_train_tok = write_bin(train_examples, os.path.join(here, 'train.bin'))
n_val_tok = write_bin(val_examples, os.path.join(here, 'val.bin'))
with open(os.path.join(here, 'test.txt'), 'w') as f:
    f.write('\n'.join(test_examples) + '\n')

with open(os.path.join(here, 'meta.pkl'), 'wb') as f:
    pickle.dump({
        'vocab_size': vocab_size,
        'stoi': stoi,
        'itos': itos,
        'length': LENGTH,
        'filler': 'O',
        'split': 'small_half',
        'split_detail': '0-2->3-5',
        'task': 'distance_parity',
        'label_rule': 'T iff abs(index(X)-index(Y)) is even, else F',
        'train_positions': TRAIN_POSITIONS,
        'test_positions': TEST_POSITIONS,
    }, f)


def class_balance(examples):
    t = sum(1 for e in examples if e.endswith('T'))
    return t, len(examples) - t


def check_examples(examples, allowed_positions):
    allowed = set(allowed_positions)
    for e in examples:
        body, label = e.split(':')
        assert len(body) == LENGTH, e
        assert body.count('X') == 1 and body.count('Y') == 1, e
        xp, yp = body.index('X'), body.index('Y')
        assert xp in allowed and yp in allowed, e
        assert (abs(xp - yp) % 2 == 0) == (label == 'T'), e
        assert all(c in 'OXY' for c in body), e


print("=== Small X/Y even/odd distance measurement data prepared ===")
print(f"LENGTH={LENGTH} SEED={SEED}")
print(f"train/val positions: {TRAIN_POSITIONS}")
print(f"test positions     : {TEST_POSITIONS}")
print(f"vocab_size={vocab_size} chars={[repr(c) for c in VOCAB_CHARS]}")
print()
for name, examples, positions in [
    ('train', train_examples, TRAIN_POSITIONS),
    ('val', val_examples, TRAIN_POSITIONS),
    ('test', test_examples, TEST_POSITIONS),
]:
    check_examples(examples, positions)
    t, f = class_balance(examples)
    unique = len(set(examples))
    print(f"{name:5s}: {len(examples):>4d} examples | unique={unique} | T={t} F={f}")
print("label and split assertions passed")
print(f"train.bin: {n_train_tok:,} tokens | val.bin: {n_val_tok:,} tokens | test.txt: {len(test_examples)} lines")
print()
print("Examples:")
for e in train_examples[:6]:
    body, label = e.split(':')
    d = abs(body.index('X') - body.index('Y'))
    print(f"  train X@{body.index('X')} Y@{body.index('Y')} d={d} -> {label}   {e}")
for e in test_examples[:6]:
    body, label = e.split(':')
    d = abs(body.index('X') - body.index('Y'))
    print(f"  test  X@{body.index('X')} Y@{body.index('Y')} d={d} -> {label}   {e}")
