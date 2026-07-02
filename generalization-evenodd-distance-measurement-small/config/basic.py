# Config for the small even/odd distance measurement sanity experiment.
#
# Advisor 6/30 setup:
#   length 6, O/X/Y input alphabet, train positions 0..2, test positions 3..5,
#   no causal mask first. The goal is to check whether each PE behaves as expected
#   before returning to larger experiments.

out_dir = 'out'
data_dir = 'data/evenodd_distance'
eval_interval = 250
log_interval = 250

# --- positional encoding: the ONE knob meant to change across experiments ---
pos_type = 'learned'   # 'none' | 'learned' | 'sinusoidal' | 'rope' | 't5'
causal = False         # advisor sanity check starts without a causal mask
readout = 'mean_body'  # classifier readout over the six body tokens

# --- tiny model ---
n_layer = 3
n_head = 2
n_embd = 32
block_size = 6         # body length 6; ':' is not fed to the model
n_classes = 2          # F/T classifier labels
dropout = 0.0
bias = True

# --- optimizer / schedule ---
batch_size = 64
max_iters = 2000
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 25
lr_decay_iters = 500

device = 'cpu'
seed = 1337
