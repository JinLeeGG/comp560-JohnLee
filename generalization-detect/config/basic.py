# Config for the detection task on the from-scratch micro-transformer engine.
#
# This replaces the old nanoGPT-flavored config. The engine is now model.py +
# pos_encoding.py + train.py in this folder (no nanoGPT import). The single independent
# variable of the PE comparison is `pos_type`; everything else is held fixed across modes.
#
# Loaded by both train.py and evaluate.py via their inline configurator:
#     ../venv/bin/python train.py    config/basic.py
#     ../venv/bin/python evaluate.py config/basic.py

out_dir = 'out'
data_dir = 'data/detect'
eval_interval = 250
log_interval = 100

# --- positional encoding: the ONE knob meant to change across experiments ---
pos_type = 'learned'   # 'none' | 'learned' | 'sinusoidal' | 'rope' | 't5'

# --- model (kept at the Phase-0 baseline size, ~0.8M params) ---
n_layer = 4
n_head = 4
n_embd = 128
block_size = 64        # one example is 22 tokens; 64 spans it with room to spare
dropout = 0.0
bias = True

# --- optimizer / schedule (in line with the previous config) ---
batch_size = 64
max_iters = 2000
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 100
lr_decay_iters = 2000

device = 'cpu'
seed = 1337
