"""
Turn the accumulated CSV logs into the headline figures for the PE comparison.

Reads (written by evaluate.py):
  - results.csv      one aggregate row per train+eval run -> bar charts (the conclusion)
  - predictions.csv  per-example diagnostic sweep        -> per-position heatmap (the why)

Produces (PNG, 150 DPI, into out/):
  - heldout_accuracy_<split>.png   held-out acc by method, seed error bars, chance line,
                                   faded in-distribution val bars for contrast
  - heldout_perclass_<split>.png   per-class (T vs F) held-out acc -> shows label collapse
  - per_position_<split>.png       accuracy over (x_pos, y_pos) pairs, one heatmap/method

Only methods that actually have rows are plotted, in a fixed order, so the figure extends
itself automatically as sinusoidal / rope / t5 are added later.

Usage (from generalization-order/):
    ../venv/bin/python plot.py --split=half
    ../venv/bin/python plot.py --split=single_pos
"""
import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Fixed, meaningful order: absolute methods together, relative/none together.
METHOD_ORDER = ['none', 'learned', 'sinusoidal', 'rope', 't5']
LENGTH = 20
CHANCE = 50.0


def read_csv(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run evaluate.py first to populate it.")
    with open(path) as f:
        return list(csv.DictReader(f))


def latest_per_key(rows, keys):
    """Dedupe: keep the row with the latest timestamp for each `keys` tuple (handles reruns)."""
    best = {}
    for r in rows:
        k = tuple(r[x] for x in keys)
        if k not in best or r.get('timestamp', '') > best[k].get('timestamp', ''):
            best[k] = r
    return list(best.values())


def methods_present(grouped):
    return [m for m in METHOD_ORDER if m in grouped]


def mean_std(vals):
    a = np.array(vals, dtype=float)
    return a.mean(), (a.std(ddof=0) if len(a) > 1 else 0.0)


def clip_err(means, stds):
    """Asymmetric error bars clipped to [0, 100] -- accuracy can't exceed those bounds, so a
    symmetric std bar would otherwise draw above 100%. Returns a (2, N) [lower; upper] array."""
    m = np.nan_to_num(np.array(means, float))
    s = np.nan_to_num(np.array(stds, float))
    return np.vstack([np.minimum(s, m), np.minimum(s, 100 - m)])


# ----------------------------- bar charts (from results.csv) -----------------------------
def plot_bars(split, out_dir):
    rows = [r for r in read_csv('results.csv') if r['split'] == split]
    if not rows:
        print(f"[bars] no results.csv rows for split={split}; skipping")
        return
    rows = latest_per_key(rows, ['pos_type', 'split', 'seed'])

    grouped = defaultdict(lambda: {'heldout': [], 'val': [], 'T': [], 'F': []})
    for r in rows:
        g = grouped[r['pos_type']]
        g['heldout'].append(float(r['heldout_acc']) * 100)
        g['T'].append(float(r['heldout_T_acc']) * 100)
        g['F'].append(float(r['heldout_F_acc']) * 100)
        if r['val_acc'] != '':
            g['val'].append(float(r['val_acc']) * 100)
    methods = methods_present(grouped)
    nseeds = {m: len(grouped[m]['heldout']) for m in methods}

    # --- Figure 1: held-out vs val, grouped bars ---
    x = np.arange(len(methods))
    w = 0.38
    ho_mean = [mean_std(grouped[m]['heldout'])[0] for m in methods]
    ho_std = [mean_std(grouped[m]['heldout'])[1] for m in methods]
    val_mean = [mean_std(grouped[m]['val'])[0] if grouped[m]['val'] else np.nan for m in methods]
    val_std = [mean_std(grouped[m]['val'])[1] if grouped[m]['val'] else 0.0 for m in methods]

    fig, ax = plt.subplots(figsize=(1.8 * len(methods) + 3, 5))
    b1 = ax.bar(x - w / 2, ho_mean, w, yerr=clip_err(ho_mean, ho_std), capsize=5,
                color='#2c7fb8', label='held-out test')
    ax.bar(x + w / 2, val_mean, w, yerr=clip_err(val_mean, val_std), capsize=4,
           color='#bdbdbd', alpha=0.7, label='in-dist. val')
    ax.axhline(CHANCE, ls='--', color='crimson', lw=1.3, label='chance (50%)')

    for i, (xi, m) in enumerate(zip(x, methods)):
        ax.text(xi - w / 2, min(ho_mean[i] + ho_std[i] + 2.5, 103),
                f"{ho_mean[i]:.0f}", ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(xi, -7, f"n={nseeds[m]}", ha='center', va='top', fontsize=8, color='gray')

    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 105)
    ax.set_ylabel('accuracy (%)')
    ax.set_xlabel('positional encoding')
    ax.set_title(f'Held-out generalization by positional encoding — {split} split\n'
                 f'(high val + low held-out = learned the task but did not generalize)',
                 fontsize=11)
    ax.legend(loc='lower left')
    fig.tight_layout()
    p1 = os.path.join(out_dir, f'heldout_accuracy_{split}.png')
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"[bars] wrote {p1}  (methods: {methods})")

    # --- Figure 2 (secondary): per-class held-out T vs F ---
    # The collapse DIRECTION is seed-dependent (some seeds collapse to T, some to F), so the
    # mean alone is misleading -- we overlay each seed as a dot to show the bimodal spread.
    fig, ax = plt.subplots(figsize=(1.8 * len(methods) + 3, 5))
    t_vals = [grouped[m]['T'] for m in methods]
    f_vals = [grouped[m]['F'] for m in methods]
    t_mean = [mean_std(v)[0] for v in t_vals]
    t_std = [mean_std(v)[1] for v in t_vals]
    f_mean = [mean_std(v)[0] for v in f_vals]
    f_std = [mean_std(v)[1] for v in f_vals]
    ax.bar(x - w / 2, t_mean, w, yerr=clip_err(t_mean, t_std), capsize=5,
           color='#41ab5d', alpha=0.55, label='T (X before Y), mean')
    ax.bar(x + w / 2, f_mean, w, yerr=clip_err(f_mean, f_std), capsize=5,
           color='#fe9929', alpha=0.55, label='F (Y before X), mean')
    for xi, tv, fv in zip(x, t_vals, f_vals):
        ax.scatter([xi - w / 2] * len(tv), tv, color='#00441b', s=22, zorder=3,
                   label='per seed' if xi == x[0] else None)
        ax.scatter([xi + w / 2] * len(fv), fv, color='#7f2704', s=22, zorder=3)
    ax.axhline(CHANCE, ls='--', color='crimson', lw=1.3, label='chance (50%)')
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylim(0, 105)
    ax.set_ylabel('held-out accuracy (%)')
    ax.set_xlabel('positional encoding')
    ax.set_title(f'Per-class held-out accuracy — {split} split\n'
                 f'(dots = seeds; learned splits to the 0/100 extremes = collapse to one label)',
                 fontsize=11)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(out_dir, f'heldout_perclass_{split}.png')
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"[bars] wrote {p2}")


# ----------------------- per-position heatmaps (from predictions.csv) -----------------------
def split_P_for(split):
    """Read the held-out position P from results.csv for a single_pos split (e.g. 'P=12')."""
    for r in read_csv('results.csv'):
        if r['split'] == split and r['split_detail'].startswith('P='):
            return int(r['split_detail'].split('=')[1])
    return LENGTH // 2


def annotate_regions(ax, split):
    """Outline the train region vs the held-out region for this split."""
    def rect(c0, r0, wide, tall, color, label, lx, ly):
        ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), wide, tall,
                               fill=False, edgecolor=color, lw=2.2))
        ax.text(lx, ly, label, color=color, fontsize=9, fontweight='bold',
                ha='center', va='center')
    if split == 'half':
        h = LENGTH // 2
        rect(0, 0, h, h, 'black', 'train', (h - 1) / 2, (h - 1) / 2)            # 0..9 x 0..9
        rect(h, h, h, h, 'blue', 'held-out', h + (h - 1) / 2, h + (h - 1) / 2)  # 10..19 x 10..19
    elif split == 'single_pos':
        P = split_P_for(split)
        ax.add_patch(Rectangle((-0.5, P - 0.5), LENGTH, 1, fill=False, edgecolor='blue', lw=2))
        ax.add_patch(Rectangle((P - 0.5, -0.5), 1, LENGTH, fill=False, edgecolor='blue', lw=2))
        ax.text(LENGTH - 1, P, f'X@{P}', color='blue', fontsize=8, ha='right', va='center')


def plot_per_position(split, out_dir):
    rows = [r for r in read_csv('predictions.csv') if r['split'] == split]
    if not rows:
        print(f"[heatmap] no predictions.csv rows for split={split}; skipping")
        return

    # accumulate correct-sum and count per (pos_type, x_pos, y_pos), pooled over seeds
    acc_sum = defaultdict(float)
    acc_cnt = defaultdict(int)
    seen = set()
    for r in rows:
        m = r['pos_type']; seen.add(m)
        key = (m, int(r['x_pos']), int(r['y_pos']))
        acc_sum[key] += int(r['correct'])
        acc_cnt[key] += 1
    methods = [m for m in METHOD_ORDER if m in seen]

    # constrained_layout reserves space for the colorbar and suptitle so nothing overlaps
    # (mixing subplots_adjust + colorbar(ax=...) + bbox_inches='tight' caused the bar to
    # collide with the right heatmap).
    fig, axes = plt.subplots(1, len(methods), figsize=(4.8 * len(methods) + 1.6, 5.0),
                             squeeze=False, constrained_layout=True)
    cmap = plt.get_cmap('RdYlGn').copy()
    cmap.set_bad('lightgray')
    im = None
    for ax, m in zip(axes[0], methods):
        grid = np.full((LENGTH, LENGTH), np.nan)
        for xp in range(LENGTH):
            for yp in range(LENGTH):
                k = (m, xp, yp)
                if acc_cnt[k]:
                    grid[xp, yp] = acc_sum[k] / acc_cnt[k]
        im = ax.imshow(grid, vmin=0, vmax=1, cmap=cmap, origin='upper')
        annotate_regions(ax, split)
        ax.set_title(m, fontsize=11)
        ax.set_xlabel('Y position')
        ax.set_ylabel('X position')
        ax.set_xticks(range(0, LENGTH, 2))
        ax.set_yticks(range(0, LENGTH, 2))
    fig.suptitle(f'Per-position accuracy over the diagnostic sweep — {split} split\n'
                 f'(cell = accuracy with X at row, Y at col; gray diagonal = X==Y excluded)',
                 fontsize=12)
    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.046, pad=0.03)
    cbar.set_label('accuracy')
    p = os.path.join(out_dir, f'per_position_{split}.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[heatmap] wrote {p}  (methods: {methods})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='half', help="which split to plot (none|single_pos|half)")
    ap.add_argument('--out_dir', default='out')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plot_bars(args.split, args.out_dir)
    plot_per_position(args.split, args.out_dir)
