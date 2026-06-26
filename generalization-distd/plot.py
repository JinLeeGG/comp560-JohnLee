"""
Turn the accumulated CSV logs into the headline figures for the dist>=D threshold PE
comparison.

Reads (written by evaluate.py):
  - results.csv      one aggregate row per train+eval run -> bar charts (the conclusion)
  - predictions.csv  per-example diagnostic sweep        -> per-position heatmap (the why)
                                                          + accuracy-vs-separation curve

Produces (PNG, 150 DPI, into out/):
  - heldout_accuracy_<split>.png        held-out acc by method, seed error bars, chance
                                        line, faded in-distribution val bars for contrast
  - heldout_perclass_<split>.png        per-class (T=far vs F=near) held-out acc
  - per_position_<split>.png            accuracy over (x1_pos, x2_pos) pairs, one heatmap
                                        per method (upper triangle; X1 < X2). BANDS parallel
                                        to the diagonal = genuine distance; BLOCKS = a
                                        position shortcut. Read this before claiming the
                                        model reads distance.
  - accuracy_vs_separation_<split>.png  (--separation) held-out region collapsed along the
                                        diagonal: mean accuracy vs distance, one line/method,
                                        with the threshold D marked.

Only methods that actually have rows are plotted, in a fixed order, so the figures extend
themselves automatically as sinusoidal / rope / t5 are added later.

Usage (from generalization-distd/):
    ../venv/bin/python plot.py --split=half               # bar + per-class + heatmap
    ../venv/bin/python plot.py --split=half --separation  # accuracy-vs-separation curve
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
# Stable color per method, so the separation-curve lines match across regenerations.
METHOD_COLOR = {'none': '#7570b3', 'learned': '#d95f02', 'sinusoidal': '#e7298a',
                'rope': '#1b9e77', 't5': '#66a61e'}
LENGTH = 20
CHANCE = 50.0
HALF = LENGTH // 2
D = 5            # dist>=D threshold (cosmetic only here: threshold line + axis labels)


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
def plot_bars(split, out_dir, methods_filter=None, suffix=''):
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
    if methods_filter:
        methods = [m for m in methods if m in methods_filter]
    nseeds = {m: len(grouped[m]['heldout']) for m in methods}

    # --- Figure 1: held-out vs val, grouped bars ---
    x = np.arange(len(methods))
    w = 0.38
    ho_mean = [mean_std(grouped[m]['heldout'])[0] for m in methods]
    ho_std = [mean_std(grouped[m]['heldout'])[1] for m in methods]
    val_mean = [mean_std(grouped[m]['val'])[0] if grouped[m]['val'] else np.nan for m in methods]
    val_std = [mean_std(grouped[m]['val'])[1] if grouped[m]['val'] else 0.0 for m in methods]

    fig, ax = plt.subplots(figsize=(1.8 * len(methods) + 3, 5))
    ax.bar(x - w / 2, ho_mean, w, yerr=clip_err(ho_mean, ho_std), capsize=5,
           color='#2c7fb8', label='held-out test')
    ax.bar(x + w / 2, val_mean, w, yerr=clip_err(val_mean, val_std), capsize=4,
           color='#bdbdbd', alpha=0.7, label='in-dist. val')
    ax.axhline(CHANCE, ls='--', color='crimson', lw=1.3, label='chance (50%)')

    for i in range(len(methods)):
        ax.text(x[i] - w / 2, min(ho_mean[i] + ho_std[i] + 2.5, 103),
                f"{ho_mean[i]:.0f}", ha='center', va='bottom', fontsize=9, fontweight='bold')

    ax.set_xticks(x)
    ax.set_xticklabels([f"{m}\n(n={nseeds[m]})" for m in methods])
    ax.set_ylim(0, 105)
    ax.set_ylabel('accuracy (%)')
    ax.set_xlabel('positional encoding')
    ax.set_title(f'dist>={D} threshold — held-out generalization by PE — {split} split\n'
                 f'(high val + low held-out = learned the task but did not generalize)',
                 fontsize=11)
    ax.legend(loc='lower left')
    fig.tight_layout()
    p1 = os.path.join(out_dir, f'heldout_accuracy_{split}{suffix}.png')
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"[bars] wrote {p1}  (methods: {methods})")

    # --- Figure 2 (secondary): per-class held-out T (far) vs F (near) ---
    # If a model collapses to a single label, per-class accuracy exposes it (one class ->
    # ~100%, the other -> ~0%). The collapse direction can be seed-dependent, so overlay
    # each seed as a dot.
    fig, ax = plt.subplots(figsize=(1.8 * len(methods) + 3, 5))
    t_vals = [grouped[m]['T'] for m in methods]
    f_vals = [grouped[m]['F'] for m in methods]
    t_mean = [mean_std(v)[0] for v in t_vals]
    t_std = [mean_std(v)[1] for v in t_vals]
    f_mean = [mean_std(v)[0] for v in f_vals]
    f_std = [mean_std(v)[1] for v in f_vals]
    ax.bar(x - w / 2, t_mean, w, yerr=clip_err(t_mean, t_std), capsize=5,
           color='#41ab5d', alpha=0.55, label=f'T (far, dist>={D}), mean')
    ax.bar(x + w / 2, f_mean, w, yerr=clip_err(f_mean, f_std), capsize=5,
           color='#fe9929', alpha=0.55, label=f'F (near, dist<{D}), mean')
    for xi, tv, fv in zip(x, t_vals, f_vals):
        ax.scatter([xi - w / 2] * len(tv), tv, color='#00441b', s=22, zorder=3,
                   label='per seed' if xi == x[0] else None)
        ax.scatter([xi + w / 2] * len(fv), fv, color='#7f2704', s=22, zorder=3)
    ax.axhline(CHANCE, ls='--', color='crimson', lw=1.3, label='chance (50%)')
    ax.set_xticks(x); ax.set_xticklabels(methods)
    ax.set_ylim(0, 105)
    ax.set_ylabel('held-out accuracy (%)')
    ax.set_xlabel('positional encoding')
    ax.set_title(f'dist>={D} threshold — per-class held-out accuracy — {split} split\n'
                 f'(dots = seeds; a class at the 0/100 extremes = collapse to one label)',
                 fontsize=11)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=8)
    fig.tight_layout()
    p2 = os.path.join(out_dir, f'heldout_perclass_{split}{suffix}.png')
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"[bars] wrote {p2}")


# ----------------------- per-position heatmaps (from predictions.csv) -----------------------
def annotate_regions(ax, split):
    """Outline the train region vs the held-out region for this split. Both X's share a
    region, so train = (first half)x(first half) and held-out = (second half)x(second half);
    the upper triangle (X1 < X2) is where the sweep has data."""
    def rect(c0, r0, wide, tall, color, label, lx, ly):
        ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), wide, tall,
                               fill=False, edgecolor=color, lw=2.2))
        ax.text(lx, ly, label, color=color, fontsize=9, fontweight='bold',
                ha='center', va='center')
    if split == 'half':
        h = HALF
        rect(0, 0, h, h, 'black', 'train', (h - 1) / 2, (h - 1) / 2)            # 0..9 x 0..9
        rect(h, h, h, h, 'blue', 'held-out', h + (h - 1) / 2, h + (h - 1) / 2)  # 10..19 x 10..19


def plot_per_position(split, out_dir, methods_filter=None, suffix=''):
    rows = [r for r in read_csv('predictions.csv') if r['split'] == split]
    if not rows:
        print(f"[heatmap] no predictions.csv rows for split={split}; skipping")
        return

    # accumulate correct-sum and count per (pos_type, x1_pos, x2_pos), pooled over seeds
    acc_sum = defaultdict(float)
    acc_cnt = defaultdict(int)
    seen = set()
    for r in rows:
        m = r['pos_type']; seen.add(m)
        key = (m, int(r['x1_pos']), int(r['x2_pos']))
        acc_sum[key] += int(r['correct'])
        acc_cnt[key] += 1
    methods = [m for m in METHOD_ORDER if m in seen]
    if methods_filter:
        methods = [m for m in methods if m in methods_filter]

    fig, axes = plt.subplots(1, len(methods), figsize=(4.8 * len(methods) + 1.6, 5.0),
                             squeeze=False, constrained_layout=True)
    cmap = plt.get_cmap('RdYlGn').copy()
    cmap.set_bad('lightgray')
    im = None
    for ax, m in zip(axes[0], methods):
        grid = np.full((LENGTH, LENGTH), np.nan)        # rows = X1 (smaller), cols = X2 (larger)
        for x1 in range(LENGTH):
            for x2 in range(LENGTH):
                k = (m, x1, x2)
                if acc_cnt[k]:
                    grid[x1, x2] = acc_sum[k] / acc_cnt[k]
        im = ax.imshow(grid, vmin=0, vmax=1, cmap=cmap, origin='upper')
        annotate_regions(ax, split)
        ax.set_title(m, fontsize=11)
        ax.set_xlabel('X2 position (larger)')
        ax.set_ylabel('X1 position (smaller)')
        ax.set_xticks(range(0, LENGTH, 2))
        ax.set_yticks(range(0, LENGTH, 2))
    fig.suptitle(f'dist>={D} threshold — per-position accuracy over the diagnostic sweep — '
                 f'{split} split\n(cell = accuracy with X at row & X at col; distance = '
                 f'col - row; BANDS // diagonal = distance, BLOCKS = position shortcut; '
                 f'gray = X1>=X2, not sampled)', fontsize=12)
    cbar = fig.colorbar(im, ax=axes[0].tolist(), fraction=0.046, pad=0.03)
    cbar.set_label('accuracy')
    p = os.path.join(out_dir, f'per_position_{split}{suffix}.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[heatmap] wrote {p}  (methods: {methods})")


# ------------------- accuracy vs separation (the collapsed held-out block) -------------------
def plot_separation(split, out_dir, methods_filter=None, suffix=''):
    """The figure from MacCormick's note: restrict the diagnostic sweep to the HELD-OUT
    region (both X's in the second half, 10..19), then for each distance value plot mean
    accuracy against distance, one line per method. This is literally the held-out block of
    the per-position heatmap collapsed along the diagonal (equal-distance cells averaged).
    The threshold D is marked: distances < D should be F (near), >= D should be T (far)."""
    rows = [r for r in read_csv('predictions.csv') if r['split'] == split]
    if not rows:
        print(f"[separation] no predictions.csv rows for split={split}; skipping")
        return

    # held-out region of the half split: both X's at positions >= HALF
    held = [r for r in rows if int(r['x1_pos']) >= HALF and int(r['x2_pos']) >= HALF]
    if not held:
        print(f"[separation] no held-out-region rows (both X's in {HALF}..{LENGTH - 1}); skipping")
        return

    # accumulate correct-sum and count per (pos_type, distance), pooled over seeds
    acc_sum = defaultdict(float)
    acc_cnt = defaultdict(int)
    seen = set()
    for r in held:
        m = r['pos_type']; seen.add(m)
        d = int(r['distance'])
        acc_sum[(m, d)] += int(r['correct'])
        acc_cnt[(m, d)] += 1
    methods = [m for m in METHOD_ORDER if m in seen]
    if methods_filter:
        methods = [m for m in methods if m in methods_filter]

    fig, ax = plt.subplots(figsize=(8, 5))
    for m in methods:
        dists = sorted(d for (mm, d) in acc_cnt if mm == m)
        ys = [100 * acc_sum[(m, d)] / acc_cnt[(m, d)] for d in dists]
        ns = [acc_cnt[(m, d)] for d in dists]
        ax.plot(dists, ys, marker='o', color=METHOD_COLOR.get(m), label=m)
        # annotate the sample count per distance once, faintly, on the first method
        if m == methods[0]:
            for d, n in zip(dists, ns):
                ax.annotate(f'n={n}', (d, -7), ha='center', va='center', fontsize=7,
                            color='gray', annotation_clip=False)

    ax.axhline(CHANCE, ls='--', color='crimson', lw=1.3, label='chance (50%)')
    # threshold line sits between the last F distance (D-1) and the first T distance (D)
    ax.axvline(D - 0.5, ls=':', color='black', lw=1.5, label=f'threshold D={D}')
    all_d = sorted({int(r['distance']) for r in held})
    ax.set_xticks(all_d)
    ax.set_ylim(0, 105)
    ax.set_xlabel(f'separation |X2 - X1|  (held-out region: both X in {HALF}..{LENGTH - 1})')
    ax.set_ylabel('mean accuracy (%)')
    ax.set_title(f'dist>={D} threshold — accuracy vs separation — {split} split\n'
                 f'(held-out block of the heatmap collapsed along the diagonal; '
                 f'dist<{D} -> F near, dist>={D} -> T far)', fontsize=11)
    ax.legend(loc='lower left', ncol=2, fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = os.path.join(out_dir, f'accuracy_vs_separation_{split}{suffix}.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[separation] wrote {p}  (methods: {methods}; distances {all_d})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='half', help="which split to plot (none|half)")
    ap.add_argument('--separation', action='store_true',
                    help="plot accuracy-vs-separation (held-out block collapsed along the diagonal)")
    ap.add_argument('--methods', default='',
                    help="comma-separated subset, e.g. none,rope (default: all present). "
                         "A subset appends '_<methods>' to the output filenames.")
    ap.add_argument('--out_dir', default='out')
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    methods_filter = [m.strip() for m in args.methods.split(',') if m.strip()] or None
    suffix = '' if not methods_filter else '_' + '-'.join(methods_filter)
    if args.separation:
        plot_separation(args.split, args.out_dir, methods_filter, suffix)
    else:
        plot_bars(args.split, args.out_dir, methods_filter, suffix)
        plot_per_position(args.split, args.out_dir, methods_filter, suffix)
