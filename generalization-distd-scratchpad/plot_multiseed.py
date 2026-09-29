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

# Publication palette: high contrast on white and distinguishable in print.
BACKGROUND = '#FFFFFF'
FOREGROUND = '#1A1A1A'
MUTED = '#5F5F5F'
GRID = '#D9D9D9'
CHANCE = '#6F6F6F'
PAIR_LINE = '#C8C8C8'
COLORS = ('#0072B2', '#D55E00', '#009E73')
MARKERS = ('o', 's', '^')


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
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100, decimals=0))
    ax.set_ylabel(ylabel)
    ax.tick_params(colors=FOREGROUND, direction='out', length=3.5, width=0.8)
    ax.grid(axis='y', color=GRID, linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(FOREGROUND)
    ax.spines['bottom'].set_color(FOREGROUND)
    ax.spines['left'].set_linewidth(0.8)
    ax.spines['bottom'].set_linewidth(0.8)


def draw_final_accuracy(ax, indexed):
    x = np.arange(len(CONDITIONS))
    jitter = np.linspace(-0.10, 0.10, len(SEEDS))
    by_condition = [
        metric_values(indexed, condition, 2000, 'heldout_free_answer_acc')
        for condition in CONDITIONS
    ]

    for seed_index, _seed in enumerate(SEEDS):
        values = [group[seed_index] for group in by_condition]
        ax.plot(
            x + jitter[seed_index],
            values,
            color=PAIR_LINE,
            linewidth=0.8,
            zorder=1,
        )

    for condition_index, (values, color) in enumerate(zip(by_condition, COLORS)):
        ax.scatter(
            condition_index + jitter,
            values,
            s=32,
            color=color,
            edgecolor=FOREGROUND,
            linewidth=0.5,
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
            markeredgecolor=FOREGROUND,
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
            fontsize=9,
        )

    ax.axhline(50, color=CHANCE, linestyle=(0, (4, 3)), linewidth=1)
    ax.set_xticks(x, CONDITION_LABELS)
    ax.set_xlabel('Condition')
    style_axis(ax, 'Unseen-position accuracy (%)')
    ax.set_title('A  Final accuracy (2,000 iterations)', loc='left', pad=30)


def draw_learning_curve(ax, indexed):
    for condition, label, color, marker in zip(
        CONDITIONS, CONDITION_LABELS, COLORS, MARKERS
    ):
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
            marker=marker,
            markersize=5.5,
            markeredgecolor=FOREGROUND,
            markeredgewidth=0.45,
            linewidth=1.8,
            capsize=3,
        )

    ax.axhline(50, color=CHANCE, linestyle=(0, (4, 3)), linewidth=1)
    ax.set_xticks(ITERATIONS)
    ax.set_xlabel('Training iteration')
    style_axis(ax, 'Mean unseen-position accuracy (%)')
    ax.set_title('B  Accuracy across checkpoints', loc='left', pad=30)
    legend = ax.legend(
        loc='lower right', frameon=False, fontsize=8.5, handlelength=2.2
    )


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
            marker=MARKERS[condition_index],
            edgecolor=FOREGROUND,
            linewidth=0.5,
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
            fontsize=9,
        )

    ax.set_xticks(x, CONDITION_LABELS)
    ax.set_xlabel('Condition')
    style_axis(ax, 'Invalid final answers (%)')
    ax.set_title('C  Invalid outputs at 2,000 iterations', loc='left', pad=30)


def main():
    args = parse_args()
    indexed = load_results(args.results)
    invalid_rates = load_invalid_answer_rates(args.predictions)
    write_summary(args.summary, indexed, invalid_rates)

    plt.rcParams.update({
        'font.family': 'DejaVu Sans',
        'font.size': 10,
        'axes.titleweight': 'bold',
        'axes.titlesize': 11,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 8.5,
        'figure.facecolor': BACKGROUND,
        'axes.facecolor': BACKGROUND,
        'text.color': FOREGROUND,
        'axes.labelcolor': FOREGROUND,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    figure, axes = plt.subplots(1, 3, figsize=(13.2, 4.7))
    draw_final_accuracy(axes[0], indexed)
    draw_learning_curve(axes[1], indexed)
    draw_invalid_answers(axes[2], invalid_rates)

    figure.suptitle(
        'Formal Scratchpad Generalization Result',
        x=0.015,
        y=0.98,
        ha='left',
        fontsize=14,
        fontweight='bold',
    )
    figure.text(
        0.015,
        0.925,
        'Five paired model seeds; NoPE; fixed input length = 20; free-running greedy inference',
        ha='left',
        color=MUTED,
        fontsize=9,
    )
    figure.text(
        0.015,
        0.025,
        'Points: individual seeds; diamond/bar: mean; error bars: sample SD; dashed line: 50% chance',
        ha='left',
        color=MUTED,
        fontsize=9,
    )
    figure.text(
        0.985,
        0.025,
        'Seen-position accuracy at 2,000 iterations: 100% for every condition and seed',
        ha='right',
        color=MUTED,
        fontsize=9,
    )
    figure.subplots_adjust(
        left=0.055,
        right=0.985,
        top=0.78,
        bottom=0.20,
        wspace=0.33,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=300, facecolor=BACKGROUND, bbox_inches='tight')
    pdf_output = args.output.with_suffix('.pdf')
    if pdf_output != args.output:
        figure.savefig(pdf_output, facecolor=BACKGROUND, bbox_inches='tight')
    plt.close(figure)
    print(f'wrote {args.summary}')
    print(f'wrote {args.output}')
    if pdf_output != args.output:
        print(f'wrote {pdf_output}')


if __name__ == '__main__':
    main()
