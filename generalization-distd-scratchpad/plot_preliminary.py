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
BACKGROUND = '#FFFFFF'
FOREGROUND = '#1A1A1A'
MUTED = '#5F5F5F'
GRID = '#D9D9D9'
CHANCE = '#6F6F6F'
SEEN_COLOR = '#0072B2'
UNSEEN_COLOR = '#D55E00'
TRUE_COLOR = '#009E73'
FALSE_COLOR = '#CC79A7'


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
        x - offset, left_means, width, color=colors[0],
        edgecolor=FOREGROUND, linewidth=0.5,
    )
    right_bars = ax.bar(
        x + offset, right_means, width, color=colors[1],
        edgecolor=FOREGROUND, linewidth=0.5, hatch='///',
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
                    facecolors='white',
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
                color=FOREGROUND,
            )

    ax.axhline(50, color=CHANCE, linestyle=(0, (4, 3)), linewidth=1)
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
            fontweight='bold',
        )

    ax.set_ylim(0, 105)
    ax.set_yticks([0, 50, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel('Final-answer accuracy (%)')
    ax.tick_params(axis='y', colors=FOREGROUND, direction='out', length=3.5)
    ax.grid(axis='y', color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(FOREGROUND)
    ax.spines['bottom'].set_color(FOREGROUND)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)
    ax.set_title(
        title,
        loc='left',
        color=FOREGROUND,
        fontsize=11,
        fontweight='bold',
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
        'axes.titleweight': 'bold',
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'figure.facecolor': BACKGROUND,
        'axes.facecolor': BACKGROUND,
        'text.color': FOREGROUND,
        'axes.labelcolor': FOREGROUND,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.8), sharey=True)

    draw_bars(
        axes[0],
        values(grouped, 'val_free_answer_acc'),
        values(grouped, 'heldout_free_answer_acc'),
        'Seen',
        'Unseen',
        (SEEN_COLOR, UNSEEN_COLOR),
        'A  Seen vs. unseen positions',
        'Positions seen vs. unseen during training',
    )

    draw_bars(
        axes[1],
        values(grouped, 'heldout_free_T_acc'),
        values(grouped, 'heldout_free_F_acc'),
        'T',
        'F',
        (TRUE_COLOR, FALSE_COLOR),
        'B  Accuracy by answer class',
        'Unseen-position accuracy · T = distance ≥ 5 · F = distance < 5',
    )
    axes[1].set_ylabel('')

    figure.suptitle(
        'Preliminary Scratchpad Generalization Result',
        x=0.015,
        y=0.98,
        ha='left',
        fontsize=14,
        fontweight='bold',
    )
    figure.text(
        0.015,
        0.915,
        'Seed 1337; 500 training iterations; NoPE; fixed input length = 20',
        ha='left',
        color=MUTED,
    )
    figure.text(
        0.015,
        0.025,
        'Free-running greedy inference; hatched bars = unseen positions / F class; dashed line = 50% chance',
        ha='left',
        color=MUTED,
        fontsize=9,
    )
    figure.text(
        0.985,
        0.025,
        'Preliminary: one model seed, no uncertainty estimate',
        ha='right',
        color=MUTED,
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.07,
        right=0.985,
        top=0.75,
        bottom=0.25,
        wspace=0.22,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, facecolor=BACKGROUND, bbox_inches='tight')
    pdf_output = args.output.with_suffix('.pdf')
    if pdf_output != args.output:
        figure.savefig(pdf_output, facecolor=BACKGROUND, bbox_inches='tight')
    plt.close(figure)
    print(f'wrote {args.output}')
    if pdf_output != args.output:
        print(f'wrote {pdf_output}')


if __name__ == '__main__':
    main()
