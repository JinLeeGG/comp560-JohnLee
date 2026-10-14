"""Paper figures for the counting experiments (Steps 2-4).

Position curves follow the standard length-generalization format (Kazemnejad et al.
2023, Fig. 3): one line per condition, a shaded band for seed variation, and a gray
region for what was seen in training.  Here the line is the mean over the five seeds
and the band is their full range (min-max), which stays inside 0-100% unlike an SD
band.  Per-seed detail goes to an appendix heatmap.  Condition colors and markers match
the distance-task figures (Okabe-Ito: no scratchpad blue circle, dummy orange square,
meaningful green triangle).  No in-figure titles; captions live in the logs.

Run from generalization-count-scratchpad/ :
    ../venv/bin/python plot_figures.py step2a_position
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'log', 'figures')
ITERATION = '2000'

# Okabe-Ito, validated (light surface): CVD worst adjacent dE 11.0, normal 25.8.
CONDITION_COLORS = {
    'no_scratchpad': '#0072B2',
    'dummy': '#D55E00',
    'meaningful': '#009E73',
}
CONDITION_MARKERS = {'no_scratchpad': 'o', 'dummy': 's', 'meaningful': '^'}
CONDITION_LABELS = {
    'no_scratchpad': 'No scratchpad',
    'dummy': 'Dummy scratchpad',
    'meaningful': 'Meaningful scratchpad',
}
INK = '#1A1A1A'
MUTED = '#6B6B6B'
GRID = '#E3E3E3'
SEEN_FILL = '#E6E6E6'

# NeurIPS text width is 5.5 in; body text is 10 pt, so figure text is 7.5-8.5 pt.
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 8,
    'axes.labelsize': 8.5,
    'xtick.labelsize': 7.5,
    'ytick.labelsize': 7.5,
    'legend.fontsize': 7.5,
    'axes.edgecolor': INK,
    'axes.labelcolor': INK,
    'xtick.color': INK,
    'ytick.color': INK,
    'axes.linewidth': 0.7,
    'xtick.major.width': 0.7,
    'ytick.major.width': 0.7,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'pdf.fonttype': 42,  # embed TrueType, not Type 3
    'savefig.dpi': 300,
})


def single_x_accuracy(predictions_csv, condition):
    """{seed: {position: accuracy %}} for inputs with exactly one X, seen and unseen."""
    hits = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    with open(predictions_csv) as file:
        for row in csv.DictReader(file):
            if (row['condition'] != condition or row['iteration'] != ITERATION
                    or row['gold'] != '1'):
                continue
            cell = hits[row['seed']][int(row['x_positions'])]
            cell[0] += row['correct'] == '1'
            cell[1] += 1
    return {
        seed: {pos: 100 * c / n for pos, (c, n) in sorted(by_pos.items())}
        for seed, by_pos in sorted(hits.items())
    }


def seed_summary(curves):
    """Positions, mean, min, and max over seeds (every seed must cover every position)."""
    positions = sorted(next(iter(curves.values())))
    for seed, by_pos in curves.items():
        if sorted(by_pos) != positions:
            raise ValueError(f'seed {seed} does not cover the same positions')
    columns = [[curves[seed][pos] for seed in curves] for pos in positions]
    return (
        positions,
        [sum(column) / len(column) for column in columns],
        [min(column) for column in columns],
        [max(column) for column in columns],
    )


def position_ticks(length, step):
    """Positions are 0..length-1, so the last tick is the last position, not length."""
    return list(range(0, length - 1, step)) + [length - 1]


def position_axes(ax, length, seen_end, label_regions=True):
    """Shared frame for position curves: seen region, grid, axes."""
    ax.axvspan(-0.5, seen_end - 0.5, color=SEEN_FILL, lw=0, zorder=0)
    ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    if label_regions:
        ax.text((seen_end - 1) / 2, 105, 'seen in training', ha='center',
                va='bottom', fontsize=7.5, color=MUTED)
        ax.text((seen_end + length - 1) / 2, 105, 'unseen (test)', ha='center',
                va='bottom', fontsize=7.5, color=MUTED)
    ax.set_xlim(-0.5, length - 0.5)
    ax.set_ylim(-4, 113)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks(position_ticks(length, 5 if length <= 20 else 10))
    ax.set_xlim(-0.5, length - 0.5)  # set_xticks may widen the view; restore it
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def position_curve(ax, curves, condition):
    """Mean over seeds (line) and the seed range (band) for one condition."""
    color = CONDITION_COLORS[condition]
    positions, mean, low, high = seed_summary(curves)
    ax.fill_between(positions, low, high, color=color, alpha=0.18, lw=0, zorder=2)
    ax.plot(positions, mean, color=color, lw=1.3, marker=CONDITION_MARKERS[condition],
            ms=3.0, mfc=color, mec='white', mew=0.5, zorder=3,
            label=CONDITION_LABELS[condition])


def position_heatmap(ax, curves, length, seen_end, label_regions=True):
    """Appendix view of the same data: one row per seed, color = accuracy."""
    seeds = list(curves)
    grid = [[curves[seed].get(pos, float('nan')) for pos in range(length)] for seed in seeds]
    image = ax.imshow(grid, aspect='auto', cmap='Greys_r', vmin=0, vmax=100,
                      interpolation='nearest',
                      extent=(-0.5, length - 0.5, len(seeds) - 0.5, -0.5))
    ax.axvline(seen_end - 0.5, color=INK, lw=1.0)
    if label_regions:
        ax.text((seen_end - 1) / 2, -0.75, 'seen in training', ha='center', va='bottom',
                fontsize=7.5, color=MUTED)
        ax.text((seen_end + length - 1) / 2, -0.75, 'unseen (test)', ha='center',
                va='bottom', fontsize=7.5, color=MUTED)
    ax.set_yticks(range(len(seeds)))
    ax.set_yticklabels([f'seed {i + 1}' for i in range(len(seeds))])
    ax.set_xticks(position_ticks(length, 10))
    ax.set_xlim(-0.5, length - 0.5)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0)
    return image


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for extension in ('pdf', 'png'):
        path = os.path.join(OUT, f'{name}.{extension}')
        fig.savefig(path, bbox_inches='tight', facecolor='white')
        print('wrote', path)
    plt.close(fig)


def load_five_seeds(predictions_csv, condition):
    curves = single_x_accuracy(os.path.join(HERE, predictions_csv), condition)
    if len(curves) != 5:
        raise ValueError(f'expected 5 seeds in {predictions_csv}, found {sorted(curves)}')
    return curves


def step2a_position():
    """Setting A (length 40, X seen in 0-19): where does the plain model miss a lone X?"""
    curves = load_five_seeds('predictions_step2A_nope.csv', 'no_scratchpad')
    fig, ax = plt.subplots(figsize=(5.5, 2.0))
    position_axes(ax, length=40, seen_end=20)
    position_curve(ax, curves, 'no_scratchpad')
    ax.set_xlabel('Position of the single X')
    ax.set_ylabel('Accuracy (%)')
    save(fig, 'step2a_position_accuracy')


def step2a_position_heatmap():
    """Appendix: the same data as step2a_position, per seed."""
    curves = load_five_seeds('predictions_step2A_nope.csv', 'no_scratchpad')
    fig, ax = plt.subplots(figsize=(5.5, 1.45))
    image = position_heatmap(ax, curves, length=40, seen_end=20)
    ax.set_xlabel('Position of the single X')
    bar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02, ticks=[0, 50, 100])
    bar.set_label('Accuracy (%)', fontsize=7.5)
    bar.ax.tick_params(labelsize=7, length=2)
    bar.outline.set_linewidth(0.5)
    save(fig, 'step2a_position_accuracy_heatmap')


FIGURES = {
    'step2a_position': step2a_position,
    'step2a_position_heatmap': step2a_position_heatmap,
}

if __name__ == '__main__':
    names = sys.argv[1:] or list(FIGURES)
    for name in names:
        FIGURES[name]()
