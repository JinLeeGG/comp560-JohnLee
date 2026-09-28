"""Prepare one raw dataset shared by all three scratchpad conditions.

Task: fixed raw length 20, exactly two X symbols, T iff their position difference
is at least 5.  Train/validation use positions 0..9; test uses 10..19.  Every
pool is class-balanced and has exactly the same distance frequencies.
"""
import os
import pickle
import random
from collections import Counter
from pathlib import Path

import numpy as np


SEED = int(os.environ.get('DATA_SEED', '1337'))
LENGTH = 20
THRESHOLD = 5
N_TRAIN = 50_000
N_VAL = 5_000
N_TEST = 2_000

DIGITS = '0123456789'
STATE_TOKENS = 'iabcdef' + 't'  # i,a,b,c,d,e,f,t
DUMMY_TOKEN = 'z'
VOCAB_CHARS = sorted(set(DIGITS + 'XTF:\n' + STATE_TOKENS + DUMMY_TOKEN))
STOI = {char: index for index, char in enumerate(VOCAB_CHARS)}
ITOS = {index: char for char, index in STOI.items()}

RNG = random.Random(SEED)
HERE = Path(__file__).resolve().parent
HALF = LENGTH // 2


def allowed_positions(role):
    if role == 'train':
        return list(range(HALF))
    if role == 'test':
        return list(range(HALF, LENGTH))
    raise ValueError(f"unknown role: {role!r}")


def split_count(total, values):
    base, remainder = divmod(total, len(values))
    return {value: base + (index < remainder) for index, value in enumerate(values)}


def positions_for_distance(allowed, distance):
    allowed_set = set(allowed)
    candidates = [(left, left + distance) for left in allowed if left + distance in allowed_set]
    return RNG.choice(candidates)


def make_example(left, right):
    chars = [RNG.choice(DIGITS) for _ in range(LENGTH)]
    chars[left] = 'X'
    chars[right] = 'X'
    distance = right - left
    label = 'T' if distance >= THRESHOLD else 'F'
    return ''.join(chars) + ':' + label


def make_pool(size, role):
    """Exactly 50/50 classes and equal distances within each class."""
    if size % 2:
        raise ValueError("pool size must be even for exact class balance")
    allowed = allowed_positions(role)
    possible_distances = list(range(1, len(allowed)))
    near = [distance for distance in possible_distances if distance < THRESHOLD]
    far = [distance for distance in possible_distances if distance >= THRESHOLD]
    per_distance = {}
    per_distance.update(split_count(size // 2, near))
    per_distance.update(split_count(size // 2, far))

    examples = []
    for distance in possible_distances:
        for _ in range(per_distance[distance]):
            left, right = positions_for_distance(allowed, distance)
            examples.append(make_example(left, right))
    RNG.shuffle(examples)
    return examples, per_distance


def validate_pool(name, examples, role, expected_distance_counts):
    labels = Counter()
    distances = Counter()
    allowed = set(allowed_positions(role))
    for example in examples:
        body, label = example.split(':')
        positions = [i for i, char in enumerate(body) if char == 'X']
        assert len(body) == LENGTH
        assert len(positions) == 2
        assert set(positions) <= allowed
        distance = positions[1] - positions[0]
        assert label == ('T' if distance >= THRESHOLD else 'F')
        labels[label] += 1
        distances[distance] += 1
    assert labels == Counter({'T': len(examples) // 2, 'F': len(examples) // 2})
    assert distances == Counter(expected_distance_counts)
    print(f"{name:5s}: {len(examples):>6,d} | T={labels['T']:,} F={labels['F']:,} "
          f"| distances={dict(sorted(distances.items()))}")


def encode(text):
    return [STOI[char] for char in text]


def write_bin(examples, path):
    stream = ''.join(example + '\n' for example in examples)
    np.asarray(encode(stream), dtype=np.uint16).tofile(path)


train_examples, train_distances = make_pool(N_TRAIN, 'train')
val_examples, val_distances = make_pool(N_VAL, 'train')
test_examples, test_distances = make_pool(N_TEST, 'test')

validate_pool('train', train_examples, 'train', train_distances)
validate_pool('val', val_examples, 'train', val_distances)
validate_pool('test', test_examples, 'test', test_distances)

write_bin(train_examples, HERE / 'train.bin')
write_bin(val_examples, HERE / 'val.bin')
write_bin(test_examples, HERE / 'test.bin')

with open(HERE / 'meta.pkl', 'wb') as file:
    pickle.dump({
        'vocab_size': len(VOCAB_CHARS),
        'stoi': STOI,
        'itos': ITOS,
        'length': LENGTH,
        'threshold': THRESHOLD,
        'split': 'half',
        'split_detail': 'first->second',
        'data_seed': SEED,
        'conditions': ('no_scratchpad', 'dummy', 'meaningful'),
    }, file)

print(f"vocab_size={len(VOCAB_CHARS)} chars={[repr(c) for c in VOCAB_CHARS]}")
print(f"wrote {HERE / 'train.bin'}, {HERE / 'val.bin'}, {HERE / 'test.bin'}")
