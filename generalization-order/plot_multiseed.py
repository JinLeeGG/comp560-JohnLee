"""
Held-out-accuracy figures for the 10-seed PE sweep, built to the dataviz skill's
principles (palette validated: relative #0072B2 / absolute #D55E00, CVD dE 91.9).

Design decisions, and the principle each follows:
  - individual seed points, no box plot  -- literature for small-n (n=10), bimodal
  - families separated by POSITION + shaded band, not colour alone  -- survives grey/CVD
  - order best->worst  -- the narrative reads left-to-right as the family collapses
  - chance line (dashed = threshold) and a faint solid val=100% reference line
    -- shows "all five LEARN (val 100%); only some GENERALIZE"
  - selective label on the one extreme (RoPE's low seed)  -- "label the extreme"
  - 2px surface ring on points  -- keeps overlapping dots legible

Two candidates:
  A' strip     -- points + mean line, no bars (most honest for bimodal `learned`)
  B' bar+points-- thin mean bars with the seed points overlaid (mean read at a glance)

Run from generalization-order/ :  ../venv/bin/python plot_multiseed.py
"""
import csv
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
CSV = os.path.join(HERE, 'results_multiseed.csv')
OUT = os.path.join(HERE, 'log', 'figures')
CHANCE = 50.0

ORDER = ['none', 'rope', 't5', 'learned', 'sinusoidal']
LABEL = {'none': 'NoPE', 'rope': 'RoPE', 't5': 'T5',
         'learned': 'Learned', 'sinusoidal': 'Sinusoidal'}
FAMILY = {'none': 'relative', 'rope': 'relative', 't5': 'relative',
          'learned': 'absolute', 'sinusoidal': 'absolute'}
FAM_COLOR = {'relative': '#0072B2', 'absolute': '#D55E00'}   # Okabe-Ito, validated
INK = '#333333'
MUTED = '#8a8a8a'


def load():
    held = defaultdict(list)
    with open(CSV) as f:
        for r in csv.DictReader(f):
            held[r['pos_type']].append(float(r['heldout_acc']) * 100)
    return held


def jitter(n, width=0.12, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(n) - 0.5) * 2 * width


def scaffold(ax, xs):
    """Family bands, chance + val reference lines, axes -- shared by both candidates."""
    # family bands (position + faint fill; labels in ink, colour carried by the marks)
    for fam in ('relative', 'absolute'):
        fx = [x for x, pe in zip(xs, ORDER) if FAMILY[pe] == fam]
        ax.axvspan(min(fx) - 0.5, max(fx) + 0.5, color=FAM_COLOR[fam], alpha=0.07, zorder=0)
        ax.text((min(fx) + max(fx)) / 2, 110, fam, ha='center', va='bottom',
                fontsize=11, color=FAM_COLOR[fam], fontweight='bold')
    # val = 100% reference (all methods): faint solid hairline (a real level, not a threshold)
    # label sits just ABOVE the line at the right, so the rule never strikes through the text
    ax.axhline(100, color=MUTED, lw=1.0, zorder=1)
    ax.text(xs[-1] + 0.72, 101.2, 'val 100%', va='bottom', ha='right', fontsize=8.5, color=MUTED)
    # chance threshold: dashed (dashing = threshold, per the mark spec)
    ax.axhline(CHANCE, ls=(0, (5, 4)), color=MUTED, lw=1.2, zorder=1)
    ax.text(xs[-1] + 0.72, 51.2, 'chance 50%', va='bottom', ha='right', fontsize=8.5, color=MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABEL[pe] for pe in ORDER])
    ax.set_ylabel('held-out accuracy (%)', color=INK)
    ax.set_ylim(0, 116)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.75, xs[-1] + 0.75)
    ax.tick_params(colors=INK)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'):
        ax.spines[s].set_color(MUTED)


def points(ax, x, v, color, size=42):
    ax.scatter(x + jitter(len(v), seed=x), v, s=size, color=color, alpha=0.8,
               edgecolor='white', linewidth=1.4, zorder=4)


def label_extreme(ax, held):
    """Selective direct label on the single narrative extreme: RoPE's low seed."""
    lo = min(held['rope'])
    xr = ORDER.index('rope')
    ax.annotate(f'{lo:.0f}', xy=(xr, lo), xytext=(xr - 0.42, lo - 2),
                fontsize=8.5, color=FAM_COLOR['relative'], va='center', ha='right',
                arrowprops=dict(arrowstyle='-', color=FAM_COLOR['relative'], lw=0.8))


def make(kind):
    held = load()
    xs = list(range(len(ORDER)))
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    scaffold(ax, xs)
    for x, pe in zip(xs, ORDER):
        v = np.array(held[pe])
        c = FAM_COLOR[FAMILY[pe]]
        if kind == 'bar':
            ax.bar(x, v.mean(), width=0.5, color=c, alpha=0.18, zorder=2,
                   edgecolor=c, linewidth=1.0)
        else:  # strip: mean line
            ax.hlines(v.mean(), x - 0.26, x + 0.26, color=c, lw=2.8, zorder=3)
        points(ax, x, v, c)
    label_extreme(ax, held)
    ax.set_title('Held-out generalization by positional encoding (10 seeds)',
                 color=INK, fontsize=12)
    fig.tight_layout()
    name = 'heldout_strip_multiseed.png' if kind == 'strip' else 'heldout_bar_multiseed.png'
    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote', p)


def heatmap():
    """Per-position accuracy, one panel per method, AVERAGED over all 10 seeds.
    Averaging (not one cherry-picked seed) is what makes `learned`'s held-out block
    read as muddy = unreliable, which is the honest story. Colour: diverging around
    chance (0.5) with a neutral white midpoint -- red = below chance (confidently
    wrong), blue = above (correct). Red/blue is the CVD-safe diverging pair (the old
    red-yellow-green map fails colour-blindness).
    """
    import numpy as np
    P = os.path.join(HERE, 'predictions_multiseed.csv')
    L = 20
    corr = {pe: np.full((L, L), np.nan) for pe in ORDER}
    acc = {pe: defaultdict(lambda: [0, 0]) for pe in ORDER}   # (x,y) -> [correct, total]
    with open(P) as f:
        for r in csv.DictReader(f):
            pe = r['pos_type']
            if pe not in acc:
                continue
            a = acc[pe][(int(r['x_pos']), int(r['y_pos']))]
            a[0] += int(r['correct']); a[1] += 1
    for pe in ORDER:
        for (x, y), (c, t) in acc[pe].items():
            corr[pe][x, y] = c / t

    # diverging map: blue = correct (top), gray = chance (neutral midpoint, per the
    # palette spec), red = below chance. blue<->red is the CVD-safe diverging pair
    # (validated); green<->red would fail colour-blindness.
    from matplotlib.colors import LinearSegmentedColormap
    div = LinearSegmentedColormap.from_list(
        'blu_red', ['#8c1d1d', '#d06b6b', '#f0efec', '#6b9fd0', '#0072B2'])
    box = dict(boxstyle='round,pad=0.2', fc='white', ec='none', alpha=0.75)

    fig, axes = plt.subplots(1, len(ORDER), figsize=(3.0 * len(ORDER), 3.4),
                             constrained_layout=True)
    im = None
    for ax, pe in zip(axes, ORDER):
        M = np.ma.masked_invalid(corr[pe])
        im = ax.imshow(M, cmap=div, vmin=0, vmax=1, origin='upper')
        ax.set_facecolor('0.80')                     # excluded diagonal shows as grey
        ax.set_title(LABEL[pe], fontsize=11, color=FAM_COLOR[FAMILY[pe]], fontweight='bold')
        ax.set_xlabel('position of Y', fontsize=8.5)
        if pe == ORDER[0]:
            ax.set_ylabel('position of X', fontsize=8.5)
        ax.tick_params(labelsize=7)
        # region separators: recessive hairlines, one step off the surface
        ax.axhline(9.5, color='0.6', lw=0.6); ax.axvline(9.5, color='0.6', lw=0.6)
        # region labels centred in each block, off the diagonal, on a white pad so
        # they stay legible over any cell colour
        ax.text(7.5, 2.0, 'trained\npositions', ha='center', va='center',
                fontsize=7.5, color='0.15', bbox=box)
        ax.text(12.0, 17.5, 'held-out\npositions', ha='center', va='center',
                fontsize=7.5, color='0.15', bbox=box)
    cbar = fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01)
    cbar.set_label('accuracy', fontsize=8.5)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(['0\n(wrong)', '0.5\n(chance)', '1\n(correct)'])
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle('Per-position accuracy by encoding, averaged over 10 seeds',
                 fontsize=12)
    p = os.path.join(OUT, 'per_position_multiseed.png')
    fig.savefig(p, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('wrote', p)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    make('bar')        # Figure 1 (chosen). make('strip') draws the rejected alternative.
    heatmap()          # Figure 2
