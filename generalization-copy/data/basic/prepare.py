"""
Based closely on Karpathy Shakespeare character-level prepare.py.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.

Copy task: model learns to copy a short lowercase string after a delimiter.
Vocab includes uppercase letters even though training data uses only lowercase,
so we can later test/fine-tune on uppercase without changing the vocab.
Example data:
abc:abc
hello:hello
xyz:xyz
"""
import os
import pickle
import random
import numpy as np

WORD_LEN_MIN = 3
WORD_LEN_MAX = 6

# vocab contains all the letters (lowercase + uppercase + delimiter + newline)
vocab_chars = sorted(list(set(
    'abcdefghijklmnopqrstuvwxyz' + 
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + 
    ':\n'
)))

vocab_size = len(vocab_chars)
stoi = {ch: i for i, ch in enumerate(vocab_chars)}
itos = {i: ch for i, ch in enumerate(vocab_chars)}

# training data uses only lowercase
alphabet = [c for c in 'abcdefghijklmnopqrstuvwxyz']
target_length = 1_000_000  # about 1MB will be plenty of training data

total_length = 0
blocks = []
while total_length < target_length:
    length = random.randint(WORD_LEN_MIN, WORD_LEN_MAX)
    word = ''.join(random.choices(alphabet, k=length))
    block = word + ':' + word
    blocks.append(block)
    total_length += len(block) + 1  # +1 for newline

data = '\n'.join(blocks)

print("First 20 lines of data:")
lines = data.split('\n')
for i in range(20):
    print(lines[i])

print(f"length of dataset in characters: {len(data):,}")
print(f"vocab size: {vocab_size:,} (includes uppercase even though data is lowercase only)")
print(f"all vocab characters: |{'|'.join(map(repr, vocab_chars))}|")

def encode(s):
    return [stoi[c] for c in s]

def decode(l):
    return ''.join([itos[i] for i in l])

# 90% train, 10% val
n = len(data)
train_data = data[:int(n * 0.9)]
val_data = data[int(n * 0.9):]

train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"train has {len(train_ids):,} tokens")
print(f"val has {len(val_ids):,} tokens")

# export to bin files
train_ids = np.array(train_ids, dtype=np.uint16)
val_ids = np.array(val_ids, dtype=np.uint16)
train_ids.tofile(os.path.join(os.path.dirname(__file__), 'train.bin'))
val_ids.tofile(os.path.join(os.path.dirname(__file__), 'val.bin'))

# save the meta information as well, to help us encode/decode later
meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(os.path.dirname(__file__), 'meta.pkl'), 'wb') as f:
    pickle.dump(meta, f)