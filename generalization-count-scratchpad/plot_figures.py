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
import matplotlib.lines  # noqa: E402
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


def step2b_position():
    """Setting B (length 40, X seen in 0-9): same format as step2a_position."""
    curves = load_five_seeds('predictions_step2B_nope.csv', 'no_scratchpad')
    fig, ax = plt.subplots(figsize=(5.5, 2.0))
    position_axes(ax, length=40, seen_end=10)
    position_curve(ax, curves, 'no_scratchpad')
    ax.set_xlabel('Position of the single X')
    ax.set_ylabel('Accuracy (%)')
    save(fig, 'step2b_position_accuracy')


def final_rows(results_csv, condition='no_scratchpad'):
    with open(os.path.join(HERE, results_csv)) as file:
        rows = [row for row in csv.DictReader(file)
                if row['iteration'] == ITERATION and row['condition'] == condition]
    if len(rows) != 5:
        raise ValueError(f'expected 5 seeds in {results_csv}, found {len(rows)}')
    return sorted(rows, key=lambda row: int(row['seed']))


def jitter(n, width=0.16):
    # Fixed, evenly spread offsets so the five seed dots never depend on randomness.
    return [width * (2 * i / (n - 1) - 1) for i in range(n)] if n > 1 else [0.0]


def step2b_settings():
    """Which split makes the plain model fail? One bar per split, chosen split highlighted."""
    settings = [
        ('results_step1_nope.csv', 'Length 20\ntrain on 0\u20139', False),
        ('results_step2A_nope.csv', 'Length 40\ntrain on 0\u201319', False),
        ('results_step2B_nope.csv', 'Length 40\ntrain on 0\u20139', True),
    ]
    blue = CONDITION_COLORS['no_scratchpad']
    fig, ax = plt.subplots(figsize=(3.6, 2.4))
    # The line stops short of its label so the text never sits on the dashes.
    ax.hlines(70, -0.55, 2.33, color=MUTED, lw=0.8, ls=(0, (4, 3)), zorder=1)
    ax.text(2.37, 70, 'pre-set\nbar (70%)', ha='left', va='center', fontsize=7,
            color=MUTED)
    for x, (csv_name, _, chosen) in enumerate(settings):
        values = [100 * float(row['heldout_acc_k1plus']) for row in final_rows(csv_name)]
        mean = sum(values) / len(values)
        ax.bar(x, mean, width=0.6, color=blue, alpha=0.85 if chosen else 0.22, lw=0,
               zorder=2)
        ax.scatter([x + dx for dx in jitter(len(values), width=0.14)], values, s=10,
                   color=INK if chosen else blue, edgecolor='white', linewidth=0.4,
                   zorder=4)
        ax.text(x, 4, f'{mean:.1f}%', ha='center', va='bottom', fontsize=8,
                color='white' if chosen else INK, fontweight='bold', zorder=5)
    ax.set_xticks(range(len(settings)))
    ax.set_xticklabels([label for _, label, _ in settings], fontsize=7.3)
    ax.set_xlim(-0.55, len(settings) - 0.05)
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel('Unseen-position accuracy (%)')
    ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', length=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    save(fig, 'step2b_settings_comparison')


CONDITIONS = ('no_scratchpad', 'dummy', 'meaningful')
# The pre-set main metric comes first; the per-count panels break it down.
COUNT_PANELS = [('acc_k1plus', 'All (1-3 X)'), ('acc_k1', '1 X'), ('acc_k2', '2 X'),
                ('acc_k3', '3 X')]


def condition_panels(axes, results_csv):
    """One panel per count: x = scratchpad condition, bar = mean, dots = seeds."""
    data = {condition: final_rows(results_csv, condition) for condition in CONDITIONS}
    for ax, (key, title) in zip(axes, COUNT_PANELS):
        for x, condition in enumerate(CONDITIONS):
            color = CONDITION_COLORS[condition]
            values = [100 * float(row[f'heldout_{key}']) for row in data[condition]]
            mean = sum(values) / len(values)
            ax.bar(x, mean, width=0.62, color=color, alpha=0.25, lw=0, zorder=2)
            ax.hlines(mean, x - 0.31, x + 0.31, color=color, lw=1.4, zorder=3)
            ax.scatter([x + dx for dx in jitter(len(values))], values, s=9, color=color,
                       marker=CONDITION_MARKERS[condition], edgecolor='white',
                       linewidth=0.4, zorder=4)
            ax.text(x, 3, f'{mean:.0f}', ha='center', va='bottom', fontsize=7,
                    color=INK, fontweight='bold', zorder=5)
        ax.set_title(title, fontsize=8, color=INK, pad=4)
        ax.set_xticks([])
        ax.set_xlim(-0.6, len(CONDITIONS) - 0.4)
        ax.set_ylim(0, 105)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='x', length=0)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)


def condition_legend(fig, y=1.0):
    handles = [
        matplotlib.lines.Line2D([], [], color=CONDITION_COLORS[c], marker=CONDITION_MARKERS[c],
                                lw=0, ms=5, mec='white', mew=0.5, label=CONDITION_LABELS[c])
        for c in CONDITIONS
    ]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, y), ncol=3,
               frameon=False, handletextpad=0.3, columnspacing=1.6)


def scratchpad_main(results_csv, name):
    fig, axes = plt.subplots(1, 4, figsize=(5.5, 2.0), sharey=True)
    condition_panels(axes, results_csv)
    axes[0].set_ylabel('Unseen-position accuracy (%)')
    fig.tight_layout(w_pad=0.6)
    condition_legend(fig)
    save(fig, name)


def scratchpad_position(predictions_csv, name):
    """One row per condition in the B-1 format, so the seed bands never overlap."""
    fig, axes = plt.subplots(len(CONDITIONS), 1, figsize=(5.5, 3.9), sharex=True)
    for row, (ax, condition) in enumerate(zip(axes, CONDITIONS)):
        position_axes(ax, length=40, seen_end=10, label_regions=(row == 0))
        position_curve(ax, load_five_seeds(predictions_csv, condition), condition)
        # Inside the seen region, where every curve sits at 100%, so it covers no data.
        ax.text(0.3, 45, CONDITION_LABELS[condition], ha='left', va='center',
                fontsize=7.5, color=INK)
    axes[-1].set_xlabel('Position of the single X')
    fig.supylabel('Accuracy on single-X inputs (%)', fontsize=8.5, x=0.01)
    fig.tight_layout(h_pad=0.4)
    save(fig, name)


ERROR_TYPES = [
    ('missed', 'Missed the +1 at an X', '#4a3aa7'),
    ('dropped', 'Dropped the count after an X', '#e87ba4'),
    ('answer', 'Count written right, answer wrong', '#eda100'),
]


def classify_trace_error(body, written):
    """Where a wrong meaningful-scratchpad answer first goes wrong."""
    count = 0
    for index, token in enumerate(body):
        count += token == 'X'
        if written[index] != str(count):
            if not written[index].isdigit() or int(written[index]) > count:
                return 'other'
            return 'missed' if token == 'X' else 'dropped'
    return 'answer'


def meaningful_error_rates(predictions_csv):
    """{seed: {error type: % of unseen inputs with 1-3 X}} at the final checkpoint."""
    counts = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    with open(os.path.join(HERE, predictions_csv)) as file:
        for row in csv.DictReader(file):
            if (row['condition'] != 'meaningful' or row['split'] != 'test'
                    or row['iteration'] != ITERATION or row['gold'] == '0'):
                continue
            totals[row['seed']] += 1
            if row['correct'] != '1':
                counts[row['seed']][classify_trace_error(row['body'], row['pred_states'])] += 1
    return {
        seed: {kind: 100 * counts[seed][kind] / totals[seed]
               for kind in ('missed', 'dropped', 'answer', 'other')}
        for seed in sorted(totals)
    }


def scratchpad_errors(predictions_csv, name):
    rates = meaningful_error_rates(predictions_csv)
    if any(by_kind['other'] for by_kind in rates.values()):
        raise ValueError(f'unclassified meaningful errors: {rates}')
    seeds = list(rates)
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    bottoms = [0.0] * len(seeds)
    for kind, label, color in ERROR_TYPES:
        heights = [rates[seed][kind] for seed in seeds]
        ax.bar(range(len(seeds)), heights, bottom=bottoms, width=0.6, color=color,
               edgecolor='white', linewidth=0.6, label=label, zorder=2)
        bottoms = [b + h for b, h in zip(bottoms, heights)]
    ax.set_xticks(range(len(seeds)))
    ax.set_xticklabels([f'seed {i + 1}' for i in range(len(seeds))], fontsize=7)
    ax.set_ylabel('Wrong answers (%)')
    ax.grid(axis='y', color=GRID, lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis='x', length=0)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.legend(loc='upper left', bbox_to_anchor=(1.02, 1.0), frameon=False,
              fontsize=7, handlelength=1.2)
    save(fig, name)


def step3_main():
    scratchpad_main('results_step3_nope.csv', 'step3_nope_accuracy_by_count')


def step3_position():
    scratchpad_position('predictions_step3_nope.csv', 'step3_nope_position_accuracy')


def step3_errors():
    scratchpad_errors('predictions_step3_nope.csv', 'step3_nope_meaningful_errors')


FIGURES = {
    'step2a_position': step2a_position,
    'step2a_position_heatmap': step2a_position_heatmap,
    'step2b_position': step2b_position,
    'step2b_settings': step2b_settings,
    'step3_main': step3_main,
    'step3_position': step3_position,
    'step3_errors': step3_errors,
}

if __name__ == '__main__':
    names = sys.argv[1:] or list(FIGURES)
    for name in names:
        FIGURES[name]()
