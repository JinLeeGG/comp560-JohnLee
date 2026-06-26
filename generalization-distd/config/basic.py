# Config for the dist>=D threshold task (distance task #2) on the from-scratch
# micro-transformer engine.
#
# Same engine as generalization-evenodd / generalization-order / generalization-detect
# (model.py + pos_encoding.py + train.py in this folder); only the data and this config are
# task-specific. The single independent variable of the PE comparison is `pos_type`;
# everything else is held fixed.
#
# Loaded by both train.py and evaluate.py via their inline configurator:
#     ../venv/bin/python train.py    config/basic.py
#     ../venv/bin/python evaluate.py config/basic.py

out_dir = 'out'
data_dir = 'data/distd'
eval_interval = 250
log_interval = 100

# --- positional encoding: the ONE knob meant to change across experiments ---
# dist>=D (T iff the two X's are >= 5 apart) depends on the MAGNITUDE of the gap, not its
# parity and not the order. Hypothesis: on the held-out 'half' split, 'none' (NoPE) may
# FAIL toward chance -- the causal mask gives order but not exact distance -- while
# distance-aware ('rope','t5') and absolute ('learned','sinusoidal', which can subtract
# positions) methods hold up. BUT dist>=D is coarse, so NoPE may also approximate it; that
# would itself be a finding (coarse distance survives NoPE, precise distance/parity does not).
pos_type = 'learned'   # 'none' | 'learned' | 'sinusoidal' | 'rope' | 't5'

# --- model (kept at the Phase-0 baseline size, ~0.8M params) ---
n_layer = 4
n_head = 4
n_embd = 128
block_size = 64        # one example is 22 tokens; 64 spans it with room to spare
dropout = 0.0
bias = True

# --- optimizer / schedule (in line with the evenodd/order/detection config) ---
batch_size = 64
max_iters = 2000
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 100
lr_decay_iters = 2000

device = 'cpu'
seed = 1337
