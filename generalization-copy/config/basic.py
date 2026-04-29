# config for copy task - phase 1 (lowercase only)

out_dir = 'out'
eval_interval = 250
eval_iters = 50
log_interval = 100

dataset = 'basic'
data_dir = 'data/basic'

# small model - enough for a simple copy task
n_layer = 4
n_head = 4
n_embd = 128
block_size = 32
dropout = 0.0

batch_size = 64
max_iters = 2000
lr_decay_iters = 2000
learning_rate = 1e-3
min_lr = 1e-4

device = 'cpu'
compile = False