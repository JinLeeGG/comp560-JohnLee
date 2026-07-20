"""
A1 diagnostic: visualize the learned T5 relative-distance bias, one figure per checkpoint,
side by side. This answers the *first* question in the readout debate — "did the PE even learn
to read distance?" — before we blame anything downstream.

WHAT IT SHOWS
-------------
T5's only positional signal is a learned scalar added to the attention logits as a function of
the *relative distance* between query and key (per head). `T5RelativeBias` stores that as an
Embedding of shape (num_buckets, n_head); for the short inputs here every distance 0..length-1
maps to its own exact bucket (the log-spaced buckets only start at distance 16), so the bias is
essentially `rel_bias.weight[distance]` per head. This script un-buckets that table and plots

    bias  vs  distance-back-to-key (0,1,2,...,length-1),  one line per head.

The task label is "T iff X and Y are distance 1 apart", so a model that reads the gap should show
a *distinctive* bias at distance 1 (marked). Compare a generalizing seed against a failing one:

  - both seeds show a clean distance-1 feature  -> the PE learned the gap in BOTH, so what
    separates 100% from 50% is NOT the PE -> the cause is downstream (readout / attention
    routing). This is the readout hypothesis's supporting evidence.
  - the failing seed's curve is flat/featureless -> the PE itself failed -> look at the PE, not
    the readout.

This is a FIRST CUT, not proof (see the plan): it cannot separate a readout problem from an
attention-routing problem. The decisive test is the mean-pool readout swap.

CHECKPOINTS
-----------
train.py always writes `<out_dir>/ckpt.pt` and overwrites it every run, so generate the two
seeds into separate dirs first (no cp needed -- out_dir is overridable):

    ../venv/bin/python data/background/prepare.py --b=1 --length=6
    ../venv/bin/python train.py config/basic.py --seed=1337 --out_dir=out_s1337
    ../venv/bin/python train.py config/basic.py --seed=1338 --out_dir=out_s1338
    ../venv/bin/python plot_t5_bias.py \
        --ckpts=out_s1337/ckpt.pt,out_s1338/ckpt.pt \
        --labels=seed1337,seed1338 --out=log/figures/t5_bias_b1_len6.png

(1337 generalizes to 100% held-out, 1338 collapses to 50% -- see results_nocolon.csv.)
"""
import os
import sys
from ast import literal_eval

import torch

from model import MicroTransformer, MicroTransformerConfig
from pos_encoding import T5RelativeBias

# ----------------------------- config (overridable) -----------------------------
ckpts = 'out/ckpt.pt'      # comma-separated checkpoint paths, plotted left-to-right
labels = ''                # comma-separated labels (defaults to the ckpt dir names)
out = 'log/figures/t5_bias.png'
max_dist = 0               # 0 = auto (length-1 from each ckpt); else cap the x-axis here
device = 'cpu'
# --------------------------------------------------------------------------------

# poor-man's configurator (same pattern as train.py / evaluate.py): --key=value overrides.
for arg in sys.argv[1:]:
    assert arg.startswith('--') and '=' in arg, f"expected --key=value, got {arg!r}"
    key, val = arg[2:].split('=', 1)
    assert key in globals(), f"unknown config key: {key}"
    try:
        val = literal_eval(val)
    except (SyntaxError, ValueError):
        pass
    globals()[key] = val

ckpt_paths = [p for p in str(ckpts).split(',') if p]
label_list = [s for s in str(labels).split(',') if s] if labels else []
assert ckpt_paths, "no checkpoints given (--ckpts=path1,path2)"
if label_list:
    assert len(label_list) == len(ckpt_paths), \
        f"got {len(label_list)} labels for {len(ckpt_paths)} ckpts"


def load_t5_bias(path):
    """Load a checkpoint, rebuild the model, and return (per-head bias vs distance, meta).

    Returns:
        dists   : list of distances-back-to-key, 0..maxd
        bias    : tensor (len(dists), n_head) -- the learned bias at each distance per head
        meta    : dict with length / b / seed / val_acc for the subplot title
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    margs = ckpt['model_args']
    assert margs.get('pos_type') == 't5', \
        f"{path} is pos_type={margs.get('pos_type')!r}, not 't5' -- this script plots the T5 bias"
    model = MicroTransformer(MicroTransformerConfig(**margs))
    model.load_state_dict(ckpt['model'])
    model.eval()
    pe = model.pe
    assert isinstance(pe, T5RelativeBias) and hasattr(pe, 'rel_bias')

    length = ckpt.get('length', margs['block_size'])
    maxd = (length - 1) if max_dist == 0 else max_dist
    dists = list(range(maxd + 1))
    # "key d positions back" => relative_position (key - query) = -d. Bucket it exactly the way
    # the model does at attention time, so this reflects the real lookup, not a re-derivation.
    rel = torch.tensor([-d for d in dists])
    buckets = T5RelativeBias._bucket(rel, pe.num_buckets, pe.max_distance, pe.causal)
    bias = pe.rel_bias.weight.detach()[buckets]            # (len(dists), n_head)
    meta = {'length': length, 'b': ckpt.get('b'), 'seed': ckpt.get('seed'),
            'val_acc': ckpt.get('val_acc')}
    return dists, bias, meta


series = [load_t5_bias(p) for p in ckpt_paths]

# ---- quantitative summary (so "clean distance-1 feature" is a number, not just a picture) ----
# distance-1 salience per head = bias(d=1) - mean(bias over d>=2). Large & positive = the head
# singles out adjacency (the T case).
print("=== T5 distance-1 salience  [ bias(d=1) - mean(bias, d>=2) ], per head ===")
for path, (dists, bias, meta) in zip(ckpt_paths, series):
    lab = f"seed {meta['seed']}"
    va = f"val {meta['val_acc']:.0%}" if meta['val_acc'] is not None else "val ?"
    d1 = dists.index(1) if 1 in dists else None
    far = [i for i, d in enumerate(dists) if d >= 2]
    print(f"\n{path}  ({lab}, {va}, length {meta['length']}, b={meta['b']})")
    if d1 is None or not far:
        print("  (need distances >= 2 to compute salience)")
        continue
    for h in range(bias.size(1)):
        sal = bias[d1, h].item() - bias[far, h].mean().item()
        print(f"  head {h}: bias(d=1)={bias[d1, h].item():+.3f}  salience={sal:+.3f}")

# --------------------------------- figure ---------------------------------
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

n = len(series)
fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.8), sharey=True, squeeze=False)
axes = axes[0]
for ax, path, (dists, bias, meta) in zip(axes, ckpt_paths, series):
    for h in range(bias.size(1)):
        ax.plot(dists, bias[:, h].tolist(), marker='o', label=f'head {h}')
    ax.axvline(1, color='0.5', ls='--', lw=1)
    ax.annotate('adjacency (T)', xy=(1, ax.get_ylim()[1]), xytext=(1.15, 0.92),
                textcoords='axes fraction', fontsize=8, color='0.4')
    ax.axhline(0, color='0.85', lw=0.8, zorder=0)
    if label_list:
        title = label_list[ckpt_paths.index(path)]
    else:
        va = f", val {meta['val_acc']:.0%}" if meta['val_acc'] is not None else ""
        title = f"seed {meta['seed']}{va}"
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('distance back to key')
    ax.set_xticks(dists)
axes[0].set_ylabel('learned T5 bias')
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle('T5 relative-distance bias per head (a spike at d=1 = the gap is read)', fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.96))

os.makedirs(os.path.dirname(out) or '.', exist_ok=True)
fig.savefig(out, dpi=150, bbox_inches='tight')
print(f"\nsaved -> {out}")
