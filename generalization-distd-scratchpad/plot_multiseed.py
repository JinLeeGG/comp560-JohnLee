"""Summarize and plot the formal five-seed scratchpad experiment."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


HERE = Path(__file__).resolve().parent
CONDITIONS = ('no_scratchpad', 'dummy', 'meaningful')
CONDITION_LABELS = ('No scratchpad', 'Dummy tokens', 'Meaningful states')
SEEDS = (1337, 1338, 1339, 1340, 1341)
ITERATIONS = (500, 1000, 1500, 2000)

BACKGROUND = '#191919'
FOREGROUND = '#F4F4F4'
MUTED = '#A7A7A7'
GRID = '#363636'
CHANCE = '#D0D0D0'
COLORS = ('#78A9D8', '#DF9254', '#69BD82')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--results', type=Path, default=HERE / 'results_multiseed.csv'
    )
    parser.add_argument(
        '--predictions', type=Path, default=HERE / 'predictions_multiseed.csv'
    )
    parser.add_argument(
        '--summary',
        type=Path,
        default=HERE / 'results_multiseed_summary.csv',
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=HERE / 'figures' / 'formal-multiseed-result.png',
    )
    return parser.parse_args()


def load_results(path):
    with path.open(newline='') as file:
        rows = list(csv.DictReader(file))

    indexed = {}
    for row in rows:
        key = (row['condition'], int(row['seed']), int(row['iteration']))
        if key in indexed:
            raise ValueError(f'duplicate result row: {key}')
        indexed[key] = row

    expected = {
        (condition, seed, iteration)
        for condition in CONDITIONS
        for seed in SEEDS
        for iteration in ITERATIONS
    }
    missing = sorted(expected - set(indexed))
    extra = sorted(set(indexed) - expected)
    if missing or extra:
        raise ValueError(f'incomplete result grid; missing={missing}, extra={extra}')
    return indexed


def load_invalid_answer_rates(path):
    counts = defaultdict(lambda: [0, 0])
    with path.open(newline='') as file:
        for row in csv.DictReader(file):
            if int(row['iteration']) != 2000:
                continue
            key = (row['condition'], int(row['seed']), row['split'])
            counts[key][0] += row['answer_pred'] not in {'T', 'F'}
            counts[key][1] += 1

    rates = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            for split in ('val', 'test'):
                key = (condition, seed, split)
                invalid, total = counts[key]
                if total == 0:
                    raise ValueError(f'missing final predictions: {key}')
                rates[key] = invalid / total
    return rates


def metric_values(indexed, condition, iteration, metric):
    return [
        100.0 * float(indexed[(condition, seed, iteration)][metric])
        for seed in SEEDS
    ]


def write_summary(path, indexed, invalid_rates):
    fields = [
        'condition', 'iteration', 'n_seeds',
        'seen_answer_mean_pct', 'seen_answer_sample_sd_pct',
        'unseen_answer_mean_pct', 'unseen_answer_sample_sd_pct',
        'unseen_T_mean_pct', 'unseen_F_mean_pct',
        'unseen_trace_exact_mean_pct',
        'unseen_invalid_answer_mean_pct',
        'unseen_invalid_answer_sample_sd_pct',
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for condition in CONDITIONS:
            for iteration in ITERATIONS:
                seen = metric_values(
                    indexed, condition, iteration, 'val_free_answer_acc'
                )
                unseen = metric_values(
                    indexed, condition, iteration, 'heldout_free_answer_acc'
                )
                unseen_t = metric_values(
                    indexed, condition, iteration, 'heldout_free_T_acc'
                )
                unseen_f = metric_values(
                    indexed, condition, iteration, 'heldout_free_F_acc'
                )
                trace_raw = [
                    indexed[(condition, seed, iteration)][
                        'heldout_trace_exact_acc'
                    ]
                    for seed in SEEDS
                ]
                trace = (
                    [100.0 * float(value) for value in trace_raw]
                    if all(trace_raw)
                    else []
                )
                invalid = (
                    [
                        100.0 * invalid_rates[(condition, seed, 'test')]
                        for seed in SEEDS
                    ]
                    if iteration == 2000
                    else []
                )
                writer.writerow({
                    'condition': condition,
                    'iteration': iteration,
                    'n_seeds': len(SEEDS),
                    'seen_answer_mean_pct': f'{mean(seen):.4f}',
                    'seen_answer_sample_sd_pct': f'{stdev(seen):.4f}',
                    'unseen_answer_mean_pct': f'{mean(unseen):.4f}',
                    'unseen_answer_sample_sd_pct': f'{stdev(unseen):.4f}',
                    'unseen_T_mean_pct': f'{mean(unseen_t):.4f}',
                    'unseen_F_mean_pct': f'{mean(unseen_f):.4f}',
                    'unseen_trace_exact_mean_pct': (
                        f'{mean(trace):.4f}' if trace else ''
                    ),
                    'unseen_invalid_answer_mean_pct': (
                        f'{mean(invalid):.4f}' if invalid else ''
                    ),
                    'unseen_invalid_answer_sample_sd_pct': (
                        f'{stdev(invalid):.4f}' if invalid else ''
                    ),
                })


def style_axis(ax, ylabel):
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel(ylabel, color=FOREGROUND)
    ax.tick_params(colors=MUTED)
    ax.grid(axis='y', color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_color(GRID)
        spine.set_linewidth(1)


def draw_final_accuracy(ax, indexed):
    x = np.arange(len(CONDITIONS))
    jitter = np.linspace(-0.10, 0.10, len(SEEDS))
    by_condition = [
        metric_values(indexed, condition, 2000, 'heldout_free_answer_acc')
        for condition in CONDITIONS
    ]

    for seed_index, _seed in enumerate(SEEDS):
        values = [group[seed_index] for group in by_condition]
        ax.plot(x + jitter[seed_index], values, color=MUTED, alpha=0.35, linewidth=0.8)

    for condition_index, (values, color) in enumerate(zip(by_condition, COLORS)):
        ax.scatter(
            condition_index + jitter,
            values,
            s=32,
            color=color,
            edgecolor=BACKGROUND,
            linewidth=0.7,
            zorder=3,
        )
        average = mean(values)
        spread = stdev(values)
        ax.errorbar(
            condition_index,
            average,
            yerr=spread,
            fmt='D',
            markersize=6,
            color=FOREGROUND,
            markerfacecolor=color,
            markeredgecolor=BACKGROUND,
            capsize=4,
            linewidth=1.2,
            zorder=4,
        )
        ax.text(
            condition_index,
            1.02,
            f'{average:.1f} ± {spread:.1f}%',
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='bottom',
            color=FOREGROUND,
            fontsize=9,
        )

    ax.axhline(50, color=CHANCE, linestyle=(0, (4, 3)), linewidth=1, alpha=0.8)
    ax.set_xticks(x, CONDITION_LABELS)
    ax.set_xlabel('Experimental condition', color=FOREGROUND)
    style_axis(ax, 'Unseen-position answer accuracy')
    ax.set_title('A. Final result at 2,000 iterations', loc='left', pad=32)


def draw_learning_curve(ax, indexed):
    for condition, label, color in zip(CONDITIONS, CONDITION_LABELS, COLORS):
        averages = []
        spreads = []
        for iteration in ITERATIONS:
            values = metric_values(
                indexed, condition, iteration, 'heldout_free_answer_acc'
            )
            averages.append(mean(values))
            spreads.append(stdev(values))
        ax.errorbar(
            ITERATIONS,
            averages,
            yerr=spreads,
            label=label,
            color=color,
            marker='o',
            markersize=5,
            linewidth=1.8,
            capsize=3,
        )

    ax.axhline(50, color=CHANCE, linestyle=(0, (4, 3)), linewidth=1, alpha=0.8)
    ax.set_xticks(ITERATIONS)
    ax.set_xlabel('Training iteration', color=FOREGROUND)
    style_axis(ax, 'Mean unseen-position accuracy')
    ax.set_title('B. Generalization across checkpoints', loc='left', pad=32)
    legend = ax.legend(
        loc='lower right', frameon=False, labelcolor=FOREGROUND, fontsize=8.5
    )
    for text in legend.get_texts():
        text.set_color(FOREGROUND)


def draw_invalid_answers(ax, invalid_rates):
    x = np.arange(len(CONDITIONS))
    jitter = np.linspace(-0.10, 0.10, len(SEEDS))
    for condition_index, (condition, color) in enumerate(zip(CONDITIONS, COLORS)):
        values = [
            100.0 * invalid_rates[(condition, seed, 'test')]
            for seed in SEEDS
        ]
        ax.scatter(
            condition_index + jitter,
            values,
            s=34,
            color=color,
            edgecolor=BACKGROUND,
            linewidth=0.7,
            zorder=3,
        )
        average = mean(values)
        ax.plot(
            [condition_index - 0.16, condition_index + 0.16],
            [average, average],
            color=FOREGROUND,
            linewidth=2,
            zorder=4,
        )
        ax.text(
            condition_index,
            1.02,
            f'mean {average:.1f}%',
            transform=ax.get_xaxis_transform(),
            ha='center',
            va='bottom',
            color=FOREGROUND,
            fontsize=9,
        )

    ax.set_xticks(x, CONDITION_LABELS)
    ax.set_xlabel('Experimental condition', color=FOREGROUND)
    style_axis(ax, 'Invalid final answers (not T or F)')
    ax.set_title('C. Why dummy accuracy falls below chance', loc='left', pad=32)


def main():
    args = parse_args()
    indexed = load_results(args.results)
    invalid_rates = load_invalid_answer_rates(args.predictions)
    write_summary(args.summary, indexed, invalid_rates)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titleweight': 'semibold',
        'figure.facecolor': BACKGROUND,
        'axes.facecolor': BACKGROUND,
        'text.color': FOREGROUND,
        'axes.labelcolor': FOREGROUND,
    })
    figure, axes = plt.subplots(1, 3, figsize=(15, 5.9))
    draw_final_accuracy(axes[0], indexed)
    draw_learning_curve(axes[1], indexed)
    draw_invalid_answers(axes[2], invalid_rates)

    figure.suptitle(
        'Formal Scratchpad Generalization Result',
        x=0.015,
        y=0.975,
        ha='left',
        fontsize=16,
        fontweight='semibold',
        color=FOREGROUND,
    )
    figure.text(
        0.015,
        0.93,
        'Five paired model seeds · NoPE · fixed input length 20 · same raw examples',
        ha='left',
        color=MUTED,
    )
    figure.text(
        0.015,
        0.035,
        'Free-running greedy inference · dots = individual seeds · diamond/line = mean · error bars = sample SD · dashed = 50% chance',
        ha='left',
        color=MUTED,
        fontsize=9,
    )
    figure.text(
        0.985,
        0.035,
        'Seen-position accuracy at 2,000 iterations: 100% for every condition and seed',
        ha='right',
        color=MUTED,
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.78,
        bottom=0.18,
        wspace=0.28,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=200, facecolor=BACKGROUND)
    plt.close(figure)
    print(f'wrote {args.summary}')
    print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
