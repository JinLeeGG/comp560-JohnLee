# Config for the distance-1 task ("is X immediately before Y?") on the from-scratch
# micro-transformer engine.
#
# Same engine as generalization-distd / evenodd / order / detect (model.py + pos_encoding.py
# + train.py in this folder); only the data and this config are task-specific. Here the
# single independent variable is NOT the positional encoding -- it is `b`, the background
# diversity, which lives in the data (data/background/prepare.py --b=N). `pos_type` is pinned
# to t5 by design: this is a T5 stress test, not a PE comparison.
#
# Loaded by both train.py and evaluate.py via their inline configurator:
#     ../venv/bin/python train.py    config/basic.py --seed=1337
#     ../venv/bin/python evaluate.py config/basic.py --seed=1337

out_dir = 'out'
data_dir = 'data/background'
eval_interval = 100      # denser than the length-20 tasks: this model converges fast, and
log_interval = 100       # the val curve is what gates Stage 1

# --- positional encoding: FIXED at t5 for this study ---
# The earlier tasks (order / evenodd / distd) all used a busy b=10 background and T5 kept
# underperforming. This experiment asks whether that was a background problem, so T5 is the
# one method under test and the background knob does the varying. ('rope' is worth a single
# spot-check if a result looks T5-specific -- see the README -- but the focus stays t5.)
pos_type = 't5'          # 'none' | 'learned' | 'sinusoidal' | 'rope' | 't5'
causal = False #jmac            # True = decoder mask; False = bidirectional attention ablation
t5_bias_mode = 'auto'    # 'auto' follows causal; use 'causal' for mask-only T5 ablation

# --- model: deliberately SMALL (MacCormick's 6/30 note) ---
# Length 6 is short and the task is easy, so the length-20 micro model (~0.80M) would be
# oversized -- it could hit 100% at every b and hide the background effect entirely. This is
# ~0.03M params, which keeps headroom limited enough for a b effect to show, and is fast
# enough for the many (b x seed) runs of Stage 2.
n_layer = 3
n_head = 2
n_embd = 32
# block_size must cover one example: LENGTH + ':' + label. Length 6 -> 8 tokens (the model
# sees 7 of them and predicts the 8th). GROW THIS IN STAGE 3: length 8 -> 10, 10 -> 12,
# 12 -> 14. train.py asserts it is big enough and says what to set.
block_size = 8
dropout = 0.0
bias = True

# --- optimizer / schedule (unchanged from the prior task folders) ---
batch_size = 64
max_iters = 500         # a small model on an easy task converges fast; read the val curve
learning_rate = 1e-3     # (curves.csv / plot.py --curve) and cut this if it plateaus early
min_lr = 1e-4
warmup_iters = 100
lr_decay_iters = 2000

device = 'cpu'
seed = 1337
