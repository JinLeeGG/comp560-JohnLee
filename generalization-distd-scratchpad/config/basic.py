# NoPE scratchpad pilot for the fixed-length dist>=5 task.

out_dir = 'out'
data_dir = 'data/distd_scratchpad'
condition = 'meaningful'  # 'no_scratchpad' | 'dummy' | 'meaningful'

eval_interval = 250
log_interval = 100

# Model: fixed across all conditions.
pos_type = 'none'
causal = True
t5_bias_mode = 'auto'
n_layer = 4
n_head = 4
n_embd = 128
block_size = 64
dropout = 0.0
bias = True

# Optimizer and fixed training budget.
batch_size = 64
max_iters = 2000
learning_rate = 1e-3
min_lr = 1e-4
warmup_iters = 100
lr_decay_iters = 2000
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.99
grad_clip = 1.0

device = 'cpu'
seed = 1337
