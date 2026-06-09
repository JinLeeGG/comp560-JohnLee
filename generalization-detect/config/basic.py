# config for the detection task - Phase 0 baseline (train on the full position distribution)
#
# Adapted from the copy task's config/basic.py. Only a few things change for detection:
#   - data_dir points at the detection dataset (vocab_size=15 is read from its meta.pkl)
#   - block_size only needs to hold one example (20 digits + ':' + label + '\n' = 23 tokens)
# Model size (n_layer/n_head/n_embd) is kept small; we shrink it further in Phase 4.

out_dir = 'out'
eval_interval = 250
eval_iters = 50
log_interval = 100

dataset = 'detect'
data_dir = 'data/detect'

# small model - more than enough for a binary detection task
n_layer = 4
n_head = 4
n_embd = 128
block_size = 64        # one example is 23 tokens; 64 comfortably spans a full example + context
dropout = 0.0

batch_size = 64
gradient_accumulation_steps = 1   # nanoGPT defaults to 40 (huge for CPU); 1 keeps each iter fast
max_iters = 2000       # detection is easier than copy; watch the eval curve, this may be plenty
lr_decay_iters = 2000
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 100

device = 'cpu'
compile = False
