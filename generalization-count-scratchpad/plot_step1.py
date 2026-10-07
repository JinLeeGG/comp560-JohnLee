"""Step 1 figure: unseen-position accuracy by count for the NoPE no-scratchpad baseline.

Same style as the relative-order poster figure (generalization-order/plot_multiseed.py):
mean bars with every seed as a point, a faint solid line for the seen-position level,
a dashed chance line, and a direct label only on the extreme seed.

Run from generalization-count-scratchpad/ :  ../venv/bin/python plot_step1.py
"""
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, 'results_step1_nope.csv')
OUT = os.path.join(HERE, 'log', 'figures')
ITERATION = '2000'
CHANCE = 25.0  # four possible answers

# Same validated blue as the poster's relative family.
BLUE = '#0072B2'
INK = '#333333'
MUTED = '#8a8a8a'

GROUPS = [
    ('acc_k1', '1 X'),
    ('acc_k2', '2 X'),
    ('acc_k3', '3 X'),
    ('acc_k1plus', 'all (1-3 X)'),
]


def load():
    with open(CSV) as file:
        rows = [row for row in csv.DictReader(file) if row['iteration'] == ITERATION]
    rows.sort(key=lambda row: int(row['seed']))
    if len(rows) != 5:
        raise ValueError(f"expected 5 seeds at iteration {ITERATION}, found {len(rows)}")
    values = {
        key: np.array([float(row[f'heldout_{key}']) * 100 for row in rows])
        for key, _ in GROUPS
    }
    seen = np.array([float(row['val_acc_all']) * 100 for row in rows])
    return values, seen, [row['seed'] for row in rows]


def jitter(n, width=0.11, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.random(n) - 0.5) * 2 * width


def main():
    values, seen, seeds = load()
    if not np.all(seen == 100.0):
        raise ValueError(f"seen-position accuracy is not 100% for every seed: {seen}")

    xs = [0, 1, 2, 3.4]  # small gap before the combined bar
    fig, ax = plt.subplots(figsize=(7.6, 4.6))

    # Bands: per-count bars vs the combined main metric (labels in ink-coloured text).
    ax.axvspan(-0.5, 2.5, color=BLUE, alpha=0.06, zorder=0)
    ax.text(1, 110, 'by number of X', ha='center', va='bottom', fontsize=11,
            color=BLUE, fontweight='bold')
    ax.axvspan(2.9, 3.9, color=MUTED, alpha=0.10, zorder=0)
    ax.text(3.4, 110, 'main metric', ha='center', va='bottom', fontsize=11,
            color=INK, fontweight='bold')

    # Reference levels: seen positions (a real level, solid) and chance (dashed).
    ax.axhline(100, color=MUTED, lw=1.0, zorder=1)
    ax.text(3.95, 101.2, 'seen positions 100%', va='bottom', ha='right',
            fontsize=8.5, color=MUTED)
    ax.axhline(CHANCE, ls=(0, (5, 4)), color=MUTED, lw=1.2, zorder=1)
    ax.text(3.95, CHANCE + 1.2, 'chance 25%', va='bottom', ha='right',
            fontsize=8.5, color=MUTED)

    for x, (key, _) in zip(xs, GROUPS):
        v = values[key]
        ax.bar(x, v.mean(), width=0.5, color=BLUE, alpha=0.32, zorder=2,
               edgecolor=BLUE, linewidth=1.0)
        ax.scatter(x + jitter(len(v), seed=int(x * 10)), v, s=42, color=BLUE,
                   alpha=0.85, edgecolor='white', linewidth=1.4, zorder=4)

    # Direct label on the single extreme: the lowest seed on a lone X.
    low_index = int(values['acc_k1'].argmin())
    low = values['acc_k1'][low_index]
    ax.annotate(f'{low:.1f} (seed {seeds[low_index]})', xy=(0, low),
                xytext=(0.42, low + 7), fontsize=8.5, color=BLUE, va='center',
                ha='left',
                arrowprops=dict(arrowstyle='-', color=BLUE, lw=0.8))

    ax.set_xticks(xs)
    ax.set_xticklabels([
        f'{label}\nmean {values[key].mean():.1f}%' for key, label in GROUPS
    ])
    ax.set_ylabel('unseen-position accuracy (%)', color=INK)
    ax.set_ylim(0, 116)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlim(-0.75, 4.0)
    ax.tick_params(colors=INK)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(MUTED)
    ax.set_title('Counting X at unseen positions, NoPE without scratchpad (5 seeds)',
                 color=INK, fontsize=12)

    fig.tight_layout()
    os.makedirs(OUT, exist_ok=True)
    for extension in ('png', 'pdf'):
        path = os.path.join(OUT, f'step1_unseen_by_count.{extension}')
        fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        print('wrote', path)
    plt.close(fig)


if __name__ == '__main__':
    main()
