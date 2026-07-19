"""
Turn the accumulated CSV logs into the figures for the background-diversity
study.

Reads (written by train.py and evaluate.py):
  - results.csv      one aggregate row per train+eval run -> accuracy-vs-b (the conclusion)
  - predictions.csv  per-example diagnostic sweep        -> per-position heatmaps (the why)
  - curves.csv       the val curve of every run          -> Stage 1's learnability gate

Produces (PNG, 150 DPI, into --out_dir):
  - run_grid_b<N>_<split>.png        (--grid) every run as one labelled square, rows = input
                                     length. Counts and individual values read off the same
                                     picture, and the SHAPE of a row shows at a glance whether
                                     the runs split into two clumps or sit on a continuum.
                                     This is the figure the Stage 1 log uses.
  - accuracy_vs_b_<split>.png        (--vs_b) x = b (background diversity), y = accuracy, two
    (len{N} for lengths != 6)        lines: in-distribution val and held-out. Read it as:
                                       both high and flat     -> background is NOT T5's problem
                                       held-out drops only    -> background hurts GENERALIZATION
                                       both drop              -> background hurts LEARNING
                                     Waiting on the sweep actually being run.
  - accuracy_vs_length_b<N>_<split>.png  (--vs_length) accuracy and outcome mix against input
                                     length. Superseded by --grid for the log, kept because it
                                     shows the per-length distribution on a common axis.
  - decay_with_distance_len<N>_b<N>.png  (--decay) error rate against how far a test case sits
                                     beyond the training region, one line per X-Y distance.
  - per_position_b<N>_<split>.png    accuracy over (x_pos, y_pos) pairs for one b, ONE PANEL
                                     PER SEED plus a pooled panel. The per-seed panels are
                                     not decoration: a pooled average can manufacture
                                     structure that no individual run shows (e.g. seeds
                                     collapsing to opposite labels average into a clean-
                                     looking pattern), so mechanism claims must be read off
                                     the per-seed panels.
  - learnability_b<N>_<split>.png    (--curve) val accuracy vs iteration, one line per seed.
                                     Shows how fast training converges and any instability
                                     afterwards. Also shows whether max_iters is wastefully
                                     long.

Usage (from generalization-background/):
    ../venv/bin/python plot.py --split=half --b=1 --grid --out_dir=log/figures
    ../venv/bin/python plot.py --split=half --b=1 --vs_length
    ../venv/bin/python plot.py --split=half --vs_b
    ../venv/bin/python plot.py --split=half --b=1 --length=12 --decay
    ../venv/bin/python plot.py --split=half                    # per-position heatmaps
    ../venv/bin/python plot.py --curve --b=1 --length=6 --split=half
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
        ax.set_title(f'distance-1 task (is Y right after X?) — accuracy vs background '
                     f'diversity\nT5, length {length}, {split} split; mean over seeds, '
                     f'error bars = std', fontsize=11)
        ax.grid(alpha=0.25)
        ax.legend(loc='lower left', fontsize=9)
        fig.tight_layout()
        p = os.path.join(out_dir, out_name or vs_b_filename(split, length))
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[vs_b] wrote {p}  (length={length}, b={bs}, seeds={nseeds})")


# --------------------- accuracy vs input length (from results.csv) ---------------------
def plot_vs_length(split, out_dir, b=1, out_name=''):
    """Held-out accuracy against input length, at fixed b — the Stage 1 gate re-run.

    Two panels, because "how well does it do?" and "how OFTEN does it work?" are different
    questions, and the mean alone actively misleads on the second one: at length 6 the mean of
    55% describes no actual run, since every seed landed on either 50% or 100%.

      left  — one dot per seed, so the bimodality at short lengths is visible directly.
      right — a stacked bar of what happened to each seed, in three outcome bands. This is
              where the real trend lives: total collapse goes 9/10 -> 0/10 across the lengths
              tested, while the perfect-score count stays flat.

    No error bars: a standard deviation drawn on a two-spike distribution suggests a spread of
    typical outcomes that does not exist. The dots carry that information honestly."""
    rows = [r for r in read_csv(RESULTS_CSV)
            if r['split'] == split and int(r['b']) == b]
    if not rows:
        print(f"[vs_length] no {RESULTS_CSV} rows for split={split}, b={b}; skipping")
        return
    rows = latest_per_key(rows, ['pos_type', 'split', 'length', 'b', 'seed'])

    grouped = defaultdict(lambda: {'heldout': [], 'val': []})
    for r in rows:
        g = grouped[int(r['length'])]
        g['heldout'].append(float(r['heldout_acc']) * 100)
        if r['val_acc'] != '':
            g['val'].append(float(r['val_acc']) * 100)
    lengths = sorted(grouped)
    nseeds = {L: len(grouped[L]['heldout']) for L in lengths}

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.6))
    x = np.arange(len(lengths))

    # --- left: one dot per seed ---
    ho_mean = [mean_std(grouped[L]['heldout'])[0] for L in lengths]
    for i, L in enumerate(lengths):
        vals = sorted(grouped[L]['heldout'])
        # spread ties sideways so ten seeds stacked on 50% read as ten dots, not one blob
        counts = defaultdict(int)
        xs = []
        for v in vals:
            counts[round(v, 3)] += 1
            xs.append(x[i] + 0.055 * ((counts[round(v, 3)] - 1) - (vals.count(v) - 1) / 2))
        ax.scatter(xs, vals, s=46, color=HELDOUT_COLOR, edgecolor='white', linewidth=0.8,
                   zorder=4, label='one seed' if i == 0 else None)
    ax.plot(x, ho_mean, marker='_', ms=26, mew=2.6, ls='', color='#c0392b', zorder=5,
            label='average of the 10 seeds')
    ax.axhline(CHANCE, ls='--', color='gray', lw=1.4)
    ax.text(x[0] - 0.42, CHANCE + 1.4, '50% = gives the same answer to everything',
            va='bottom', ha='left', fontsize=9, color='#555555', style='italic')
    ax.set_ylim(44, 106)
    ax.set_ylabel('accuracy in held-out positions (%)', fontsize=10)
    ax.set_title('How well did each run do?', fontsize=12, fontweight='bold')
    ax.legend(loc='center left', fontsize=9, framealpha=0.95)

    # --- right: what happened to each seed (the real trend) ---
    fail = [sum(1 for v in grouped[L]['heldout'] if v <= 50.001) for L in lengths]
    perfect = [sum(1 for v in grouped[L]['heldout'] if v >= 99.999) for L in lengths]
    partial = [nseeds[L] - f - p for L, f, p in zip(lengths, fail, perfect)]
    bars = [('failed completely\n(50% — picks one answer)', fail, '#c0392b'),
            ('partly correct', partial, '#f0a830'),
            ('perfect (100%)', perfect, '#27ae60')]
    bottom = np.zeros(len(lengths))
    for label, vals, color in bars:
        ax2.bar(x, vals, 0.62, bottom=bottom, color=color, label=label,
                edgecolor='white', linewidth=1.2)
        for xi, (v, b0) in enumerate(zip(vals, bottom)):
            if v:
                ax2.text(x[xi], b0 + v / 2, str(v), ha='center', va='center',
                         fontsize=11, fontweight='bold', color='white')
        bottom += np.array(vals, dtype=float)
    ax2.set_ylim(0, max(nseeds.values()) + 2.8)     # headroom for the in-panel legend
    ax2.set_ylabel('number of seeds (out of 10)', fontsize=10)
    ax2.set_title('What happened across the 10 runs?', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper center', ncol=3, fontsize=8.5, framealpha=0.95,
               handlelength=1.4, columnspacing=1.0)

    for a in (ax, ax2):
        a.set_xticks(x)
        a.set_xticklabels([f"length {L}" for L in lengths], fontsize=10)
        a.set_xlabel('length of the input string', fontsize=10)
        a.grid(axis='y', alpha=0.25)
        a.set_axisbelow(True)
    fig.suptitle('Does a longer input fix the problem?', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    p = os.path.join(out_dir, out_name or f'accuracy_vs_length_b{b}_{split}.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[vs_length] wrote {p}")
    for i, L in enumerate(lengths):
        print(f"    length {L:>2} (n={nseeds[L]:>2}): mean {ho_mean[i]:5.1f}% | "
              f"failed {fail[i]}/{nseeds[L]} | partial {partial[i]}/{nseeds[L]} | "
              f"perfect {perfect[i]}/{nseeds[L]}")


# --------------------- generalization decay with distance (from predictions.csv) ---------
def plot_decay(split, out_dir, length=12, b=1, out_name=''):
    """Error rate against how far a test case sits beyond the training region.

    The held-out half is not uniformly hard. Cells right next to the training boundary are
    answered almost perfectly; cells at the far end are not. Plotting error against that
    distance turns "held-out accuracy = 91%" into the more useful statement that the learned
    rule degrades gradually as it is carried further from where it was taught.

    One line per gap size, because the two effects compound: small gaps are harder than large
    ones AND far positions are harder than near ones. Only non-collapsed seeds are included —
    a seed that answers one label everywhere has 100% error at every distance and would flatten
    the curves into a meaningless straight line."""
    acc = {(int(r['length']), int(r['seed'])): float(r['heldout_acc']) * 100
           for r in read_csv(RESULTS_CSV) if r['split'] == split and int(r['b']) == b}
    rows = [r for r in read_csv(PREDICTIONS_CSV)
            if r['split'] == split and int(r['b']) == b and int(r['length']) == length]
    if not rows:
        print(f"[decay] no {PREDICTIONS_CSV} rows for length={length}, b={b}; skipping")
        return
    half = length // 2
    live = {s for (L, s) in acc if L == length and acc[(L, s)] > 50.001}
    if not live:
        print(f"[decay] every seed collapsed at length={length}; nothing to plot")
        return

    # error rate per (gap, distance-beyond-the-training-region), pooled over non-collapsed seeds
    tally = defaultdict(lambda: [0, 0])
    for r in rows:
        x_pos, gap = int(r['x_pos']), int(r['gap'])
        if x_pos < half or int(r['seed']) not in live:
            continue
        t = tally[(gap, x_pos - half)]
        t[0] += (r['correct'] == '0')
        t[1] += 1
    gaps = sorted({g for (g, _) in tally})

    # Gaps that never produce a mistake would otherwise be several lines stacked invisibly on
    # y=0; collapse them into one labelled baseline so the two lines that DO move stay legible.
    def errs(g):
        ds = sorted(d for (gg, d) in tally if gg == g)
        return ds, [100 * tally[(g, d)][0] / tally[(g, d)][1] for d in ds]

    moving = [g for g in gaps if any(y > 0 for y in errs(g)[1])]
    flat = [g for g in gaps if g not in moving]

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    colors = ['#c0392b', '#e08e0b', '#2c7fb8', '#6a51a3']
    for i, g in enumerate(moving):
        ds, ys = errs(g)
        note = '  ← X and Y nearly touching' if g == min(moving) else ''
        ax.plot(ds, ys, marker='o', ms=8, lw=2.8, color=colors[i % len(colors)],
                label=f'gap {g}{note}', zorder=3)
        for d, y in zip(ds, ys):
            if y > 0:
                ax.annotate(f'{y:.0f}%', (d, y), textcoords='offset points', xytext=(0, 9),
                            ha='center', fontsize=9, fontweight='bold',
                            color=colors[i % len(colors)])
    if flat:
        all_d = sorted({d for (_, d) in tally})
        ax.plot(all_d, [0] * len(all_d), lw=2.4, color='#2e8b57', zorder=2,
                label=f"gap {', '.join(str(g) for g in flat)} — never wrong, anywhere")

    ax.set_xticks(sorted({d for (_, d) in tally}))
    ax.set_ylim(-4, max(70, 5 + max(max(errs(g)[1]) for g in moving)))
    ax.set_xlabel('how far beyond the training region the answer sits\n'
                  '(0 = the first position the model never saw)', fontsize=10)
    ax.set_ylabel('mistakes (%)', fontsize=10)
    ax.set_title('Mistakes grow the further the answer sits from the trained positions\n'
                 f'input length {length}, clean background (b={b}), {len(live)} seeds',
                 fontsize=12, fontweight='bold')
    ax.legend(title='distance between X and Y', fontsize=9.5, title_fontsize=9.5,
              loc='upper left', framealpha=0.95)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(out_dir, out_name or f'decay_with_distance_len{length}_b{b}.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[decay] wrote {p}  (length={length}, seeds={len(live)}, gaps={gaps})")
    for g in gaps:
        ds = sorted(d for (gg, d) in tally if gg == g)
        print(f"    gap {g}: " + "  ".join(
            f"+{d}={100 * tally[(g, d)][0] / tally[(g, d)][1]:.0f}%" for d in ds))


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
            # mark the distance-1 (T) cells: y = x+1, the first off-diagonal
            for x in range(length - 1):
                ax.add_patch(Rectangle((x + 1 - 0.5, x - 0.5), 1, 1, fill=False,
                                       edgecolor='black', lw=1.0, ls=':'))
            ax.set_title(f"run {panel}" if panel != 'pooled' else 'all runs averaged',
                         fontsize=10)
            ax.set_xlabel('where Y sits')
            ax.set_ylabel('where X sits')
            ax.set_xticks(range(length))
            ax.set_yticks(range(length))
            ax.tick_params(labelsize=8)
        for ax in flat_axes[len(panels):]:
            ax.axis('off')
        fig.suptitle(f'Where does each run get it right?   '
                     f'(input length {length}, clean background b={b})\n'
                     f'green = correct, red = wrong, grey = not applicable (X must be left of Y). '
                     f'Each square is one X/Y placement;\ndotted squares are the "touching" (T) '
                     f'cases. Black box = positions used in training, blue box = unseen positions.',
                     fontsize=11)
        cbar = fig.colorbar(im, ax=flat_axes[:len(panels)].tolist(), fraction=0.035, pad=0.02)
        cbar.set_label('fraction answered correctly')
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
        ax.set_title('Learning the task is never the problem\n'
                     f'every run reaches 100% on TRAINED positions by iteration ~100  '
                     f'(length {length}, b={b}, {len(by_seed)} runs)',
                     fontsize=12, fontweight='bold')
        ax.set_ylabel('accuracy on TRAINED positions (%)', fontsize=10)
        ax.grid(alpha=0.25)
        ax.set_axisbelow(True)
        # 10 near-identical lines make a 10-entry legend useless; name only the shape that varies
        ax.plot([], [], ' ', label=f'{len(by_seed)} runs overlaid')
        ax.legend(loc='lower right', fontsize=8, ncol=2)
        fig.tight_layout()
        name = out_name or (f'learnability_b{b}_{grp_split}.png' if length == 6
                            else f'learnability_b{b}_len{length}_{grp_split}.png')
        p = os.path.join(out_dir, name)
        fig.savefig(p, dpi=150)
        plt.close(fig)
        print(f"[curve] wrote {p}  (seeds: {sorted(by_seed)})")


# --------------------- run grid: one square per run (from results.csv) ---------------------
def plot_run_grid(split, out_dir, b=1, out_name=''):
    """Every run as one labelled square: rows = input length, columns = the 10 runs.

    This is the whole dataset of this experiment in one picture. The scatter-plus-bar version
    it replaces showed the same 40 numbers twice — once as positions on an axis, once as
    counts in a stacked bar — and asked the reader to reconcile them. Here each run is a
    single cell carrying its own accuracy, so counts are read by scanning a row and values are
    read off the cell, with no cross-referencing.

    Runs are sorted within a row rather than kept in seed order: seed 1337 at length 6 and
    seed 1337 at length 12 are different data, so the column index means nothing, and sorting
    makes the shape of each row legible (a clean split at length 6, a gradient at length 12)."""
    rows = [r for r in read_csv(RESULTS_CSV) if r['split'] == split and int(r['b']) == b]
    if not rows:
        print(f"[grid] no {RESULTS_CSV} rows for split={split}, b={b}; skipping")
        return
    rows = latest_per_key(rows, ['pos_type', 'split', 'length', 'b', 'seed'])
    by_len = defaultdict(list)
    for r in rows:
        by_len[int(r['length'])].append(float(r['heldout_acc']) * 100)
    # Longest first in the list, and since row 0 is drawn at y=0 (the bottom), that puts the
    # SHORTEST length in the top row -- so the rows read 6, 8, 10, 12 downwards, matching the
    # order the log presents them in.
    lengths = sorted(by_len, reverse=True)
    ncol = max(len(v) for v in by_len.values())

    fig, ax = plt.subplots(figsize=(1.02 * ncol + 3.6, 0.92 * len(lengths) + 2.5))
    # red at chance -> green at perfect; the midpoint is meaningless here, so a plain ramp
    cmap = plt.get_cmap('RdYlGn')
    for row, L in enumerate(lengths):
        vals = sorted(by_len[L])
        for col, v in enumerate(vals):
            shade = (v - 50) / 50                    # 50% -> 0, 100% -> 1
            ax.add_patch(Rectangle((col, row), 0.92, 0.86, facecolor=cmap(0.08 + 0.84 * shade),
                                   edgecolor='white', linewidth=2))
            ax.text(col + 0.46, row + 0.43, f"{v:.0f}", ha='center', va='center',
                    fontsize=10.5, fontweight='bold',
                    color='white' if shade < 0.28 or shade > 0.72 else '#333333')
        n_fail = sum(1 for v in vals if v <= 50.001)
        n_perf = sum(1 for v in vals if v >= 99.999)
        ax.text(ncol + 0.25, row + 0.43,
                f"{n_fail} failed · {n_perf} perfect", va='center', fontsize=10, color='#444444')

    ax.set_yticks([r + 0.43 for r in range(len(lengths))])
    ax.set_yticklabels([f"length {L}" for L in lengths], fontsize=11)
    ax.set_xticks([])
    ax.set_xlim(-0.15, ncol + 3.4)
    ax.set_ylim(-0.2, len(lengths))
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_title('Accuracy in held-out positions — one square per run',
                 fontsize=13, fontweight='bold', pad=12)
    # Training success is deliberately a sentence, not a per-cell number: every run reached
    # 100% on trained positions at either iteration 100 or 200, and eval_interval is 100, so
    # the difference is a single measurement tick. Printing it per cell would imply a
    # resolution the data does not have.
    ax.text(0, -0.10,
            f'{ncol} runs per length, sorted low to high.   '
            '50 = chance, reached by answering a single label everywhere.',
            fontsize=9.5, color='#555555', va='top')
    ax.text(0, -0.34,
            f'Learning was never the problem: all {ncol * len(lengths)} runs reached 100% on '
            'the positions they were trained on, within 100–200 iterations.',
            fontsize=9.5, color='#555555', va='top', style='italic')
    fig.tight_layout()
    p = os.path.join(out_dir, out_name or f'run_grid_b{b}_{split}.png')
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print(f"[grid] wrote {p}  (lengths={lengths}, {ncol} runs each)")


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--split', default='half', help="which split to plot (none|half)")
    ap.add_argument('--vs_b', action='store_true',
                    help="plot the headline accuracy-vs-b figure (val + held-out)")
    ap.add_argument('--vs_length', action='store_true',
                    help="plot accuracy + seed-success-rate against input length (Stage 1 gate)")
    ap.add_argument('--grid', action='store_true',
                    help="plot every run as one labelled square (rows = length)")
    ap.add_argument('--decay', action='store_true',
                    help="plot error rate against distance beyond the training region")
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
    elif args.vs_length:
        plot_vs_length(args.split, args.out_dir, b=(b_filter[0] if b_filter else 1),
                       out_name=args.out_name)
    elif args.grid:
        plot_run_grid(args.split, args.out_dir, b=(b_filter[0] if b_filter else 1),
                      out_name=args.out_name)
    elif args.decay:
        plot_decay(args.split, args.out_dir,
                   length=(length_filter[0] if length_filter else 12),
                   b=(b_filter[0] if b_filter else 1), out_name=args.out_name)
    elif args.vs_b:
        plot_vs_b(args.split, args.out_dir, out_name=args.out_name)
    else:
        plot_per_position(args.split, args.out_dir, b_filter, length_filter)
