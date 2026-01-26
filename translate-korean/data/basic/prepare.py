"""
Based closely on Karpathy Shakespeare character-level prepare.py.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
Korean number translation: 일 one, 이 two, 삼 three, etc.
"""
import os
import pickle
import numpy as np
import random

# Korean number words and their English translations
korean_numbers = [
    "일 one",
    "이 two",
    "삼 three",
    "사 four",
    "오 five",
    "육 six",
    "칠 seven",
    "팔 eight",
    "구 nine",
    "십 ten",
    "십일 eleven",
    "십이 twelve",
    "십삼 thirteen",
    "십사 fourteen",
    "십오 fifteen",
    "십육 sixteen",
    "십칠 seventeen",
    "십팔 eighteen",
    "십구 nineteen",
    "이십 twenty",
]

# Create building block by repeating random selections
# Each line is a Korean-English pair
building_block = ""
for _ in range(100):  # 100 pairs per block
    pair = random.choice(korean_numbers)
    building_block += pair + "\n"

target_length = 1_000_000  # about 1MB will be plenty of training data

# construct a string consisting of repeated copies of building_block, such that it has length at least target_length
data = (building_block * ((target_length // len(building_block)) + 1))

print(f"length of dataset in characters: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(data)))
vocab_size = len(chars)
print(f"all the unique characters: |{'|'.join(map(repr,chars))}|")
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
def encode(s):
    return [stoi[c] for c in s] # encoder: take a string, output a list of integers
def decode(l):
    return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

# create the train and test splits (90/10)
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]

# encode both to integers
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
