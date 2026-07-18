"""
Turn the accumulated CSV logs into the figures for the adjacency / background-diversity
study.

Reads (written by train.py and evaluate.py):
  - results.csv      one aggregate row per train+eval run -> accuracy-vs-b (the conclusion)
  - predictions.csv  per-example diagnostic sweep        -> per-position heatmaps (the why)
  - curves.csv       the val curve of every run          -> Stage 1's learnability gate

Produces (PNG, 150 DPI, into out/):
  - accuracy_vs_b_<split>.png        THE HEADLINE. x = b (background diversity), y = accuracy,
    (len{N} in the name for      two lines: in-distribution val and held-out test, mean over
     lengths other than 6)       seeds with error bars, chance line at 50%. Read it as:
                                   both lines high and flat  -> background is NOT T5's problem
                                   held-out drops, val high   -> background hurts GENERALIZATION
                                   both drop                  -> background hurts LEARNING
  - per_position_b<N>_<split>.png    accuracy over (x_pos, y_pos) pairs for one b, ONE PANEL
                                     PER SEED plus a pooled panel. The per-seed panels are
                                     not decoration: a pooled average can manufacture
                                     structure that no individual run shows (e.g. seeds
                                     collapsing to opposite labels average into a clean-
                                     looking pattern), so mechanism claims must be read off
                                     the per-seed panels.
  - learnability_b<N>.png            (--curve) val accuracy vs iteration, one line per seed.
                                     Stage 1's gate: flat at 50% = it cannot learn the task;
                                     climbing = good. Also shows whether max_iters is
                                     wastefully long.

Usage (from generalization-adjacent/):
    ../venv/bin/python plot.py --split=half --vs_b        # headline accuracy-vs-b
    ../venv/bin/python plot.py --split=half               # per-position heatmaps (all b present)
    ../venv/bin/python plot.py --curve --b=1              # Stage 1 learnability curve
    ../venv/bin/python plot.py --split=half --vs_b --out_dir=log/figures
"""
import os
import csv
import math
import argparse
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

CHANCE = 50.0
VAL_COLOR = '#d9d9d9'        # in-distribution val   (gray, matching the prior task folders)
VAL_LINE = '#737373'
HELDOUT_COLOR = '#2c7fb8'    # held-out test         (blue,  matching the prior task folders)
RESULTS_CSV = 'results.csv'
PREDICTIONS_CSV = 'predictions.csv'
CURVES_CSV = 'curves.csv'


def read_csv(path):
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} -- run train.py/evaluate.py first to populate it.")
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


def mean_std(vals):
    a = np.array(vals, dtype=float)
    return a.mean(), (a.std(ddof=0) if len(a) > 1 else 0.0)


def clip_err(means, stds):
    """Asymmetric error bars clipped to [0, 100] -- accuracy cannot exceed those bounds, so a
    symmetric std bar would otherwise draw above 100%. Returns a (2, N) [lower; upper] array."""
    m = np.nan_to_num(np.array(means, float))
    s = np.nan_to_num(np.array(stds, float))
    return np.vstack([np.minimum(s, m), np.minimum(s, 100 - m)])


def vs_b_filename(split, length):
    """Length 6 is the debugging anchor and gets the plain name; Stage 3's longer inputs get
    the length in the filename so the figures do not overwrite each other."""
    return (f'accuracy_vs_b_{split}.png' if length == 6
            else f'accuracy_vs_b_len{length}_{split}.png')


# --------------------- THE HEADLINE: accuracy vs b (from results.csv) ---------------------
def plot_vs_b(split, out_dir, out_name=''):
    """One figure per input length: accuracy against background diversity b, with the
    in-distribution val line and the held-out line together. Plotting them together is the
    whole point -- the gap between them is what separates "never learned it" from "learned it
    but could not generalize"."""
    rows = [r for r in read_csv(RESULTS_CSV) if r['split'] == split]
    if not rows:
        print(f"[vs_b] no {RESULTS_CSV} rows for split={split}; skipping")
        return
    rows = latest_per_key(rows, ['pos_type', 'split', 'length', 'b', 'seed'])

    by_length = defaultdict(list)
    for r in rows:
        by_length[int(r['length'])].append(r)

    for length in sorted(by_length):
        grouped = defaultdict(lambda: {'heldout': [], 'val': []})
        for r in by_length[length]:
            g = grouped[int(r['b'])]
            g['heldout'].append(float(r['heldout_acc']) * 100)
            if r['val_acc'] != '':
                g['val'].append(float(r['val_acc']) * 100)
        bs = sorted(grouped)
        nseeds = {b: len(grouped[b]['heldout']) for b in bs}

        ho_mean = [mean_std(grouped[b]['heldout'])[0] for b in bs]
        ho_std = [mean_std(grouped[b]['heldout'])[1] for b in bs]
        val_mean = [mean_std(grouped[b]['val'])[0] if grouped[b]['val'] else np.nan for b in bs]
        val_std = [mean_std(grouped[b]['val'])[1] if grouped[b]['val'] else 0.0 for b in bs]

        fig, ax = plt.subplots(figsize=(9.0, 5.4))
        ax.errorbar(bs, val_mean, yerr=clip_err(val_mean, val_std), marker='s', ms=6, lw=2.0,
                    capsize=4, color=VAL_LINE, mfc=VAL_COLOR, label='in-dist. val (did it LEARN?)')
        ax.errorbar(bs, ho_mean, yerr=clip_err(ho_mean, ho_std), marker='o', ms=6.5, lw=2.4,
                    capsize=4, color=HELDOUT_COLOR, label='held-out test (did it GENERALIZE?)')
        # individual seeds behind the means: a mean of 75% from two seeds at 100/50 is a very
        # different result from four seeds all at 75%, and only the dots show which it is.
        for i, b in enumerate(bs):
            vals = grouped[b]['heldout']
            jitter = np.linspace(-0.12, 0.12, len(vals)) if len(vals) > 1 else [0]
            ax.scatter([b + j for j in jitter], vals, s=20, color='#08306b',
                       edgecolor='white', linewidth=0.5, zorder=4, alpha=0.85,
                       label='held-out, per seed' if i == 0 else None)
        ax.axhline(CHANCE, ls='--', color='crimson', lw=1.3, label='chance (50%)')

        ax.set_xticks(bs)
        ax.set_xticklabels([f"{b}\n(n={nseeds[b]})" for b in bs])
        ax.set_ylim(0, 105)
        ax.set_xlabel('b  —  number of distinct background token types')
        ax.set_ylabel('accuracy (%)')
        ax.set_title(f'adjacency (is X immediately before Y?) — accuracy vs background '
                     f'diversity\nT5, length {length}, {split} split; mean over seeds, '
                     f'error bars = std', fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(loc='lower left', fontsize=9)
        fig.tight_layout()
        p = os.path.join(out_dir, out_name or vs_b_filename(split, length))
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[vs_b] wrote {p}  (length={length}, b={bs}, seeds={nseeds})")


# --------------------- per-position heatmaps (from predictions.csv) ---------------------
def annotate_regions(ax, split, length):
    """Outline the train region vs the held-out region. X and Y share a region, so
    train = (first half) x (first half) and held-out = (second half) x (second half); only
    the upper triangle (X < Y) has data, since X is always left of Y."""
    if split != 'half':
        return
    h = length // 2

    def rect(c0, r0, side, color, label):
        ax.add_patch(Rectangle((c0 - 0.5, r0 - 0.5), side, side,
                               fill=False, edgecolor=color, lw=2.2))
        ax.text(c0 + (side - 1) / 2, r0 + (side - 1) / 2, label, color=color,
                fontsize=8, fontweight='bold', ha='center', va='center')
    rect(0, 0, h, 'black', 'train')
    rect(h, h, h, 'blue', 'held-out')


def plot_per_position(split, out_dir, b_filter=None, length_filter=None):
    """One figure per (length, b): a panel per seed, plus a pooled panel.

    How to read a panel: the T cells are exactly the first off-diagonal (y = x+1); everything
    else is F. If accuracy is uniform along that off-diagonal and uniform off it, the model is
    reading the GAP (relative distance) -- the intended solution. If accuracy instead tracks
    WHERE X or Y sits (bright/dark columns or blocks), it is a position shortcut, which at
    length 6 the training region cannot distinguish on its own. Compare panels before
    concluding: structure that appears only in the pooled panel is an averaging artifact."""
    rows = [r for r in read_csv(PREDICTIONS_CSV) if r['split'] == split]
    if not rows:
        print(f"[heatmap] no {PREDICTIONS_CSV} rows for split={split}; skipping")
        return

    groups = defaultdict(list)
    for r in rows:
        groups[(int(r['length']), int(r['b']))].append(r)

    for (length, b) in sorted(groups):
        if (b_filter and b not in b_filter) or (length_filter and length not in length_filter):
            continue
        grows = groups[(length, b)]
        seeds = sorted({int(r['seed']) for r in grows})
        panels = [str(s) for s in seeds] + (['pooled'] if len(seeds) > 1 else [])

        # accumulate correct-sum and count per (panel, x_pos, y_pos)
        acc_sum, acc_cnt = defaultdict(float), defaultdict(int)
        for r in grows:
            for key in ((r['seed'], int(r['x_pos']), int(r['y_pos'])),
                        ('pooled', int(r['x_pos']), int(r['y_pos']))):
                acc_sum[key] += int(r['correct'])
                acc_cnt[key] += 1

        ncols = min(3, len(panels))
        nrows = math.ceil(len(panels) / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.9 * ncols + 1.0, 3.85 * nrows),
                                 squeeze=False, constrained_layout=True)
        cmap = plt.get_cmap('RdYlGn').copy()
        cmap.set_bad('lightgray')
        im = None
        flat_axes = axes.ravel()
        for ax, panel in zip(flat_axes, panels):
            grid = np.full((length, length), np.nan)     # rows = X position, cols = Y position
            for x in range(length):
                for y in range(length):
                    k = (panel, x, y)
                    if acc_cnt[k]:
                        grid[x, y] = acc_sum[k] / acc_cnt[k]
            im = ax.imshow(grid, vmin=0, vmax=1, cmap=cmap, origin='upper')
            annotate_regions(ax, split, length)
            # mark the adjacency (T) cells: y = x+1, the first off-diagonal
            for x in range(length - 1):
                ax.add_patch(Rectangle((x + 1 - 0.5, x - 0.5), 1, 1, fill=False,
                                       edgecolor='black', lw=1.0, ls=':'))
            ax.set_title(f"seed {panel}" if panel != 'pooled' else 'pooled over seeds',
                         fontsize=10)
            ax.set_xlabel('Y position')
            ax.set_ylabel('X position')
            ax.set_xticks(range(length))
            ax.set_yticks(range(length))
            ax.tick_params(labelsize=8)
        for ax in flat_axes[len(panels):]:
            ax.axis('off')
        fig.suptitle(f'adjacency — per-position diagnostic sweep (b={b}, length={length}, '
                     f'{split} split)\ndotted cells (y = x+1) are the T/adjacent cases; '
                     f'uniform along the gap = distance-reading, position blocks = shortcut',
                     fontsize=11)
        cbar = fig.colorbar(im, ax=flat_axes[:len(panels)].tolist(), fraction=0.035, pad=0.02)
        cbar.set_label('accuracy')
        name = (f'per_position_b{b}_{split}.png' if length == 6
                else f'per_position_b{b}_len{length}_{split}.png')
        p = os.path.join(out_dir, name)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[heatmap] wrote {p}  (seeds: {seeds})")


# --------------------- Stage 1 learnability curve (from curves.csv) ---------------------
def plot_curve(out_dir, b_filter=None, split='', out_name='', length_filter=None):
    """Val accuracy against training iteration, one line per seed, one figure per (length, b).

    This is Stage 1's gate in visual form: a line pinned flat at 50% means the model never
    learned the task at all (stop and report), a line climbing to ~100% means it did. It also
    shows where the curve plateaus, which is how max_iters gets tuned down instead of burning
    iterations on a flat line."""
    rows = read_csv(CURVES_CSV)
    if split:
        rows = [r for r in rows if r['split'] == split]
    if not rows:
        print(f"[curve] no {CURVES_CSV} rows{' for split=' + split if split else ''}; skipping")
        return

    # Grouped by SPLIT as well as (length, b): the same b is run under both the 'half' and
    # 'none' splits, and merging them would silently drop one behind the other (identical
    # seed labels, deduped by timestamp) -- and they are different experiments.
    groups = defaultdict(list)
    for r in rows:
        groups[(r['split'], int(r['length']), int(r['b']))].append(r)

    # A single --out_name cannot name several figures: without this guard the last group
    # silently overwrites the earlier ones and the file is mislabeled.
    def keep(g):
        return ((not b_filter or g[2] in b_filter)
                and (not length_filter or g[1] in length_filter))

    wanted = [g for g in sorted(groups) if keep(g)]
    if out_name and len(wanted) > 1:
        raise SystemExit(
            f"--out_name={out_name!r} but {len(wanted)} groups match "
            f"{[(s, f'len{l}', f'b{b}') for s, l, b in wanted]}; "
            f"narrow it with --split/--b/--length.")

    for g in sorted(groups):
        if not keep(g):
            continue
        grp_split, length, b = g
        grows = groups[g]
        # keep only the latest run per (seed, iter) so reruns do not draw twice
        grows = latest_per_key(grows, ['seed', 'iter'])
        by_seed = defaultdict(list)
        for r in grows:
            by_seed[int(r['seed'])].append((int(r['iter']), float(r['val_acc']) * 100))

        fig, ax = plt.subplots(figsize=(8.2, 5.0))
        for seed in sorted(by_seed):
            pts = sorted(by_seed[seed])
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker='o', ms=3.5, lw=1.8,
                    label=f'seed {seed}')
        ax.axhline(CHANCE, ls='--', color='crimson', lw=1.3, label='chance (50%)')
        ax.set_ylim(0, 105)
        ax.set_xlabel('training iteration')
        ax.set_ylabel('in-distribution val accuracy (%)')
        ax.set_title(f'adjacency — learnability curve (b={b}, length={length}, '
                     f'{grp_split} split, T5)\n'
                     f'flat at 50% = never learned the task; climbing = learned',
                     fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(loc='lower right', fontsize=9)
        fig.tight_layout()
        name = out_name or (f'learnability_b{b}_{grp_split}.png' if length == 6
                            else f'learnability_b{b}_len{length}_{grp_split}.png')
        p = os.path.join(out_dir, name)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[curve] wrote {p}  (seeds: {sorted(by_seed)})")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='half', help="which split to plot (none|half)")
    ap.add_argument('--vs_b', action='store_true',
                    help="plot the headline accuracy-vs-b figure (val + held-out)")
    ap.add_argument('--curve', action='store_true',
                    help="plot the val-accuracy learnability curve from curves.csv")
    ap.add_argument('--b', default='',
                    help="comma-separated subset of b values for the heatmap/curve "
                         "(default: every b present)")
    ap.add_argument('--length', default='',
                    help="comma-separated subset of input lengths for the heatmap/curve "
                         "(default: every length present)")
    ap.add_argument('--out_name', default='',
                    help="override the output filename (single-figure modes only), e.g. "
                         "stage1_learnability_b1.png")
    ap.add_argument('--out_dir', default='out')
    ap.add_argument('--results_csv', default='results.csv')
    ap.add_argument('--predictions_csv', default='predictions.csv')
    ap.add_argument('--curves_csv', default='curves.csv')
    args = ap.parse_args()
    RESULTS_CSV = args.results_csv
    PREDICTIONS_CSV = args.predictions_csv
    CURVES_CSV = args.curves_csv
    os.makedirs(args.out_dir, exist_ok=True)
    b_filter = [int(x) for x in args.b.split(',') if x.strip()] or None
    length_filter = [int(x) for x in args.length.split(',') if x.strip()] or None

    if args.curve:
        plot_curve(args.out_dir, b_filter, split=args.split, out_name=args.out_name,
                   length_filter=length_filter)
    elif args.vs_b:
        plot_vs_b(args.split, args.out_dir, out_name=args.out_name)
    else:
        plot_per_position(args.split, args.out_dir, b_filter, length_filter)
