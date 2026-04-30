"""
Evaluation script for the copy task.
Generates random test inputs, asks the model to copy them, and measures accuracy.

Usage:
    NANOGPT_CONFIG=../../comp560-nanoGPT/configurator.py python -u evaluate.py config/basic.py
"""
import os
import sys
import pickle
import random
import torch

# add nanoGPT to path so we can import the model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'comp560-nanoGPT'))
from model import GPTConfig, GPT

# ----- config (can be overridden by configurator.py) -----
out_dir = 'out'
data_dir = 'data/basic'
device = 'cpu'
seed = 1337
num_test = 100              # how many test examples to evaluate
test_alphabet = 'lowercase'  # 'lowercase', 'uppercase', or 'mixed'
word_len_min = 3
word_len_max = 6
# ---------------------------------------------------------

# load configurator overrides
exec(open(os.environ.get('NANOGPT_CONFIG', 'configurator.py')).read())

random.seed(seed)
torch.manual_seed(seed)

# load meta
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
stoi = meta['stoi']
itos = meta['itos']

def encode(s):
    return [stoi[c] for c in s]

def decode(ids):
    return ''.join([itos[i] for i in ids])

# load model from checkpoint
ckpt_path = os.path.join(out_dir, 'ckpt.pt')
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
gptconf = GPTConfig(**checkpoint['model_args'])
model = GPT(gptconf)
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k, v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
model.load_state_dict(state_dict)
model.eval()
model.to(device)

# pick alphabet for testing
if test_alphabet == 'lowercase':
    alphabet = 'abcdefghijklmnopqrstuvwxyz'
elif test_alphabet == 'uppercase':
    alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
elif test_alphabet == 'mixed':
    alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
else:
    raise ValueError(f"unknown test_alphabet: {test_alphabet}")

# check that all test characters exist in vocab
missing = [c for c in alphabet if c not in stoi]
if missing:
    print(f"WARNING: these characters are not in vocab and will be skipped: {missing}")
    alphabet = ''.join([c for c in alphabet if c in stoi])

# run evaluation
correct = 0
total = 0
errors = []

for i in range(num_test):
    length = random.randint(word_len_min, word_len_max)
    word = ''.join(random.choices(alphabet, k=length))
    prompt = word + ':'
    
    prompt_ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    
    with torch.no_grad():
        # generate exactly len(word) more tokens
        output_ids = model.generate(prompt_ids, max_new_tokens=len(word), temperature=0.1, top_k=1)
    
    generated = decode(output_ids[0].tolist())
    # extract just the part after the delimiter
    predicted = generated[len(prompt):len(prompt) + len(word)]
    
    is_correct = (predicted == word)
    if is_correct:
        correct += 1
    else:
        errors.append((word, predicted))
    total += 1

accuracy = correct / total
print(f"\n=== Evaluation Results ===")
print(f"Test alphabet: {test_alphabet}")
print(f"Number of tests: {total}")
print(f"Correct: {correct}")
print(f"Accuracy: {accuracy:.2%}")

if errors:
    print(f"\n=== First 10 errors ===")
    for word, predicted in errors[:10]:
        print(f"  expected: {word:10s} got: {predicted}")