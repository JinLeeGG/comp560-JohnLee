"""Plot the preliminary scratchpad result from results_gate.csv."""
import argparse
import csv
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


HERE = Path(__file__).resolve().parent
CONDITIONS = ('no_scratchpad', 'dummy', 'meaningful')
CONDITION_LABELS = ('No\nscratchpad', 'Dummy\nscratchpad', 'Meaningful\nscratchpad')
SEEN_COLOR = '#4C78A8'
UNSEEN_COLOR = '#F28E2B'
TRUE_COLOR = '#59A14F'
FALSE_COLOR = '#B279A2'


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=HERE / 'results_gate.csv')
    parser.add_argument(
        '--output',
        type=Path,
        default=HERE / 'figures' / 'preliminary-scratchpad-result.png',
    )
    return parser.parse_args()


def load_results(path):
    grouped = {condition: [] for condition in CONDITIONS}
    with path.open(newline='') as file:
        for row in csv.DictReader(file):
            condition = row['condition']
            if condition in grouped:
                grouped[condition].append(row)
    missing = [condition for condition, rows in grouped.items() if not rows]
    if missing:
        raise ValueError(f"missing conditions in {path}: {missing}")
    return grouped


def values(grouped, metric):
    return [
        [100.0 * float(row[metric]) for row in grouped[condition]]
        for condition in CONDITIONS
    ]


def draw_bars(ax, left_values, right_values, left_label, right_label, colors):
    x = np.arange(len(CONDITIONS))
    width = 0.34
    left_means = [mean(group) for group in left_values]
    right_means = [mean(group) for group in right_values]

    left_bars = ax.bar(
        x - width / 2, left_means, width, label=left_label, color=colors[0]
    )
    right_bars = ax.bar(
        x + width / 2, right_means, width, label=right_label, color=colors[1]
    )

    for centers, groups, bars in (
        (x - width / 2, left_values, left_bars),
        (x + width / 2, right_values, right_bars),
    ):
        for center, group, bar in zip(centers, groups, bars):
            if len(group) > 1:
                offsets = np.linspace(-0.045, 0.045, len(group))
                ax.scatter(
                    center + offsets,
                    group,
                    s=22,
                    facecolors='white',
                    edgecolors='#222222',
                    linewidths=0.7,
                    zorder=3,
                )
                ax.errorbar(
                    center,
                    mean(group),
                    yerr=stdev(group),
                    color='#222222',
                    capsize=3,
                    linewidth=1,
                    zorder=4,
                )
            value = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                max(1.5, value + 2.0),
                f'{value:.0f}%',
                ha='center',
                va='bottom',
                fontsize=9,
                fontweight='semibold',
            )

    ax.axhline(50, color='#555555', linestyle=(0, (4, 3)), linewidth=1)
    ax.set_xticks(x, CONDITION_LABELS)
    ax.set_ylim(0, 122)
    ax.set_yticks([0, 50, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel('Final-answer accuracy')
    ax.set_xlabel('Experimental condition')
    ax.grid(axis='y', color='#D9D9D9', linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, loc='upper left', ncols=2)


def main():
    args = parse_args()
    grouped = load_results(args.results)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titleweight': 'semibold',
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
    })
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    draw_bars(
        axes[0],
        values(grouped, 'val_free_answer_acc'),
        values(grouped, 'heldout_free_answer_acc'),
        'Seen positions',
        'Unseen positions',
        (SEEN_COLOR, UNSEEN_COLOR),
    )
    axes[0].set_title('A. All models learn; only one generalizes', loc='left')

    draw_bars(
        axes[1],
        values(grouped, 'heldout_free_T_acc'),
        values(grouped, 'heldout_free_F_acc'),
        'T (distance ≥ 5)',
        'F (distance < 5)',
        (TRUE_COLOR, FALSE_COLOR),
    )
    axes[1].set_title(
        'B. Why the controls score 50% on unseen positions', loc='left'
    )
    axes[1].set_ylabel('')

    figure.suptitle(
        'Preliminary Scratchpad Generalization Result',
        x=0.06,
        y=0.98,
        ha='left',
        fontsize=16,
        fontweight='semibold',
    )
    figure.text(
        0.06,
        0.925,
        'Seed 1337 · 500 training iterations · NoPE · fixed input length 20',
        ha='left',
        color='#555555',
    )
    figure.text(
        0.06,
        0.015,
        'Free-running greedy inference · dashed line = chance (50%) · '
        'preliminary: one model seed, no uncertainty estimate',
        ha='left',
        color='#555555',
        fontsize=9,
    )
    figure.tight_layout(rect=(0.03, 0.07, 0.99, 0.91), w_pad=2.5)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, bbox_inches='tight')
    plt.close(figure)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
