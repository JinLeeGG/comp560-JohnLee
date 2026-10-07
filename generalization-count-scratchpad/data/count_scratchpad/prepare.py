"""Prepare one raw counting dataset shared by all scratchpad conditions.

Task: a fixed-length string of lowercase filler letters contains k copies of X
(k in 0..3).  The answer is k as one digit.  Train/validation place every X in the
seen region [0, seen_end); the held-out test places every X in the unseen region
[seen_end, length).  Every pool has exactly the same count distribution.

Default (Step 1): length 20, seen 0..9, unseen 10..19.
Longer variants change --length and --seen_end together for the whole setting;
train and test inputs always have the same length.

    ../venv/bin/python data/count_scratchpad/prepare.py
    ../venv/bin/python data/count_scratchpad/prepare.py --length=40 --seen_end=10 \
        --out_dir=data/count_scratchpad_L40_seen10
"""
import argparse
import pickle
import random
import string
from collections import Counter
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent

parser = argparse.ArgumentParser()
parser.add_argument('--length', type=int, default=20)
parser.add_argument('--seen_end', type=int, default=None,
                    help='X positions 0..seen_end-1 are seen; default is length // 2')
parser.add_argument('--max_count', type=int, default=3)
parser.add_argument('--n_train', type=int, default=50_000)
parser.add_argument('--n_val', type=int, default=5_000)
parser.add_argument('--n_test', type=int, default=2_000)
parser.add_argument('--data_seed', type=int, default=1337)
parser.add_argument('--out_dir', type=Path, default=HERE)
args = parser.parse_args()

LENGTH = args.length
SEEN_END = LENGTH // 2 if args.seen_end is None else args.seen_end
COUNTS = list(range(args.max_count + 1))
if not 0 < SEEN_END < LENGTH:
    raise ValueError(f"seen_end must be in 1..{LENGTH - 1}, got {SEEN_END}")
if args.max_count > min(SEEN_END, LENGTH - SEEN_END):
    raise ValueError("max_count must fit inside both the seen and unseen regions")
if args.max_count > 9:
    raise ValueError("answers must be single digits")

# Lowercase filler, excluding 'x' so it cannot be confused with the target X.
FILLER = ''.join(char for char in string.ascii_lowercase if char != 'x')
COUNT_TOKENS = ''.join(str(count) for count in COUNTS)
DUMMY_TOKEN = '#'
# One vocabulary for every condition so parameter counts match across conditions.
VOCAB_CHARS = sorted(set(FILLER + 'X' + COUNT_TOKENS + DUMMY_TOKEN + ':\n'))
STOI = {char: index for index, char in enumerate(VOCAB_CHARS)}
ITOS = {index: char for char, index in STOI.items()}

RNG = random.Random(args.data_seed)


def allowed_positions(role):
    if role == 'seen':
        return list(range(SEEN_END))
    if role == 'unseen':
        return list(range(SEEN_END, LENGTH))
    raise ValueError(f"unknown role: {role!r}")


def split_count(total, values):
    base, remainder = divmod(total, len(values))
    return {value: base + (index < remainder) for index, value in enumerate(values)}


def make_example(count, allowed):
    chars = [RNG.choice(FILLER) for _ in range(LENGTH)]
    for position in RNG.sample(allowed, count):
        chars[position] = 'X'
    return ''.join(chars) + ':' + str(count)


def make_pool(size, role):
    """Equal examples per count; X positions are a uniform random subset of the region."""
    allowed = allowed_positions(role)
    per_count = split_count(size, COUNTS)
    examples = [
        make_example(count, allowed)
        for count in COUNTS
        for _ in range(per_count[count])
    ]
    RNG.shuffle(examples)
    return examples, per_count


def validate_pool(name, examples, role, expected_counts):
    counts = Counter()
    position_hits = Counter()
    allowed = set(allowed_positions(role))
    for example in examples:
        body, label = example.split(':')
        positions = [i for i, char in enumerate(body) if char == 'X']
        assert len(body) == LENGTH
        assert set(body) <= set(FILLER + 'X')
        assert set(positions) <= allowed
        assert label == str(len(positions))
        counts[int(label)] += 1
        position_hits.update(positions)
    assert counts == Counter(expected_counts)
    covered = sorted(position_hits)
    print(f"{name:5s}: {len(examples):>6,d} | counts={dict(sorted(counts.items()))} "
          f"| X positions {covered[0]}..{covered[-1]} ({len(covered)} used)")


def encode(text):
    return [STOI[char] for char in text]


def write_bin(examples, path):
    stream = ''.join(example + '\n' for example in examples)
    np.asarray(encode(stream), dtype=np.uint16).tofile(path)


train_examples, train_counts = make_pool(args.n_train, 'seen')
val_examples, val_counts = make_pool(args.n_val, 'seen')
test_examples, test_counts = make_pool(args.n_test, 'unseen')

validate_pool('train', train_examples, 'seen', train_counts)
validate_pool('val', val_examples, 'seen', val_counts)
validate_pool('test', test_examples, 'unseen', test_counts)

train_set = set(train_examples)
overlap_val = sum(example in train_set for example in val_examples)
overlap_test = sum(example in train_set for example in test_examples)
assert overlap_val == 0 and overlap_test == 0, (overlap_val, overlap_test)
print("no exact duplicates between train and val/test")

args.out_dir.mkdir(parents=True, exist_ok=True)
write_bin(train_examples, args.out_dir / 'train.bin')
write_bin(val_examples, args.out_dir / 'val.bin')
write_bin(test_examples, args.out_dir / 'test.bin')

with open(args.out_dir / 'meta.pkl', 'wb') as file:
    pickle.dump({
        'vocab_size': len(VOCAB_CHARS),
        'stoi': STOI,
        'itos': ITOS,
        'length': LENGTH,
        'seen_end': SEEN_END,
        'counts': COUNTS,
        'filler': FILLER,
        'split': f'X seen in 0..{SEEN_END - 1}, unseen in {SEEN_END}..{LENGTH - 1}',
        'data_seed': args.data_seed,
        'conditions': ('no_scratchpad', 'dummy', 'meaningful'),
    }, file)

print(f"vocab_size={len(VOCAB_CHARS)} chars={''.join(VOCAB_CHARS)!r}")
print(f"wrote train/val/test .bin and meta.pkl to {args.out_dir}")
