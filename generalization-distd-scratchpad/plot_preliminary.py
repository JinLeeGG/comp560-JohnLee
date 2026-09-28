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
BACKGROUND = '#191919'
FOREGROUND = '#F4F4F4'
MUTED = '#A7A7A7'
GRID = '#363636'
CHANCE = '#D0D0D0'
SEEN_COLOR = '#78A9D8'
UNSEEN_COLOR = '#DF9254'
TRUE_COLOR = '#69BD82'
FALSE_COLOR = '#CB80AA'


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


def draw_bars(
    ax,
    left_values,
    right_values,
    left_label,
    right_label,
    colors,
    title,
    context,
):
    x = np.arange(len(CONDITIONS))
    width = 0.24
    offset = 0.14
    left_means = [mean(group) for group in left_values]
    right_means = [mean(group) for group in right_values]

    left_bars = ax.bar(
        x - offset, left_means, width, color=colors[0]
    )
    right_bars = ax.bar(
        x + offset, right_means, width, color=colors[1]
    )

    for centers, groups, bars in (
        (x - offset, left_values, left_bars),
        (x + offset, right_values, right_bars),
    ):
        for center, group, bar in zip(centers, groups, bars):
            if len(group) > 1:
                offsets = np.linspace(-0.045, 0.045, len(group))
                ax.scatter(
                    center + offsets,
                    group,
                    s=22,
                    facecolors=BACKGROUND,
                    edgecolors=FOREGROUND,
                    linewidths=0.7,
                    zorder=3,
                )
                ax.errorbar(
                    center,
                    mean(group),
                    yerr=stdev(group),
                    color=FOREGROUND,
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
                color=FOREGROUND,
            )

    ax.axhline(50, color=CHANCE, linestyle=(0, (4, 3)), linewidth=1, alpha=0.8)
    ax.set_xticks([])
    for group_index, condition_label in enumerate(CONDITION_LABELS):
        ax.text(
            group_index - offset,
            -0.045,
            left_label,
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='top',
            color=MUTED,
            fontsize=8.5,
        )
        ax.text(
            group_index + offset,
            -0.045,
            right_label,
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='top',
            color=MUTED,
            fontsize=8.5,
        )
        ax.text(
            group_index,
            -0.13,
            condition_label,
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='top',
            color=FOREGROUND,
            fontsize=9,
            fontweight='semibold',
        )

    ax.set_ylim(0, 110)
    ax.set_yticks([0, 50, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel('Final-answer accuracy (%)', color=FOREGROUND)
    ax.set_xlabel('Experimental condition', color=FOREGROUND, labelpad=73)
    ax.tick_params(axis='y', colors=MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(1)
    ax.set_title(
        title,
        loc='left',
        color=FOREGROUND,
        fontsize=12.5,
        fontweight='semibold',
        pad=31,
    )
    ax.text(
        0,
        1.025,
        context,
        transform=ax.transAxes,
        ha='left',
        va='bottom',
        color=MUTED,
        fontsize=8.5,
    )


def main():
    args = parse_args()
    grouped = load_results(args.results)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titleweight': 'semibold',
        'figure.facecolor': BACKGROUND,
        'axes.facecolor': BACKGROUND,
        'text.color': FOREGROUND,
    })
    figure, axes = plt.subplots(1, 2, figsize=(12, 6.8), sharey=True)

    draw_bars(
        axes[0],
        values(grouped, 'val_free_answer_acc'),
        values(grouped, 'heldout_free_answer_acc'),
        'Seen',
        'Unseen',
        (SEEN_COLOR, UNSEEN_COLOR),
        'A. All models learn; only one generalizes',
        'Positions seen vs. unseen during training',
    )

    draw_bars(
        axes[1],
        values(grouped, 'heldout_free_T_acc'),
        values(grouped, 'heldout_free_F_acc'),
        'T',
        'F',
        (TRUE_COLOR, FALSE_COLOR),
        'B. Why the controls score 50% on unseen positions',
        'Unseen-position accuracy · T = distance ≥ 5 · F = distance < 5',
    )
    axes[1].set_ylabel('')

    figure.suptitle(
        'Preliminary Scratchpad Generalization Result',
        x=0.015,
        y=0.965,
        ha='left',
        fontsize=16,
        fontweight='semibold',
        color=FOREGROUND,
    )
    figure.text(
        0.015,
        0.918,
        'Seed 1337 · 500 training iterations · NoPE · fixed input length 20',
        ha='left',
        color=MUTED,
    )
    figure.text(
        0.015,
        0.055,
        'Free-running greedy inference · dashed line = chance (50%)',
        ha='left',
        color=MUTED,
        fontsize=9,
    )
    figure.text(
        0.985,
        0.055,
        'Preliminary: one model seed, no uncertainty estimate',
        ha='right',
        color=MUTED,
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.72,
        bottom=0.28,
        wspace=0.2,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, facecolor=BACKGROUND)
    plt.close(figure)
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
