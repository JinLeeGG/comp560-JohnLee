"""Validate, summarize, and plot the single-token placement experiment.

The Step 5 no-scratchpad reference is reused; Step 6 adds one supplied '#' before
or after the same raw input. All three conditions have answer-only supervision.

Run from this experiment directory after the two-condition sweep completes:

    ../venv/bin/python analyze_step6.py

Result and prediction files are read-only inputs. The two output CSVs and the
publication figures are regenerated from them. Prediction diagnostics remain
local and are not intended to be committed.
"""
import argparse
import csv
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

import numpy as np


HERE = Path(__file__).resolve().parent
CONDITIONS = ('no_scratchpad', 'prefix_one', 'suffix_one')
NEW_CONDITIONS = CONDITIONS[1:]
SEEDS = (1337, 1338, 1339, 1340, 1341)
ITERATIONS = (500, 1000, 1500, 2000)
SPLIT_SIZES = {'val': 5000, 'test': 2000}
META = {'length': 40, 'seen_end': 10, 'data_seed': 1337, 'vocab_size': 43}
COLORS = ('#0072B2', '#D55E00', '#CC79A7')
MARKERS = ('o', '^', 's')
LABELS = ('No scratchpad', '# before input', '# after input')
TICK_LABELS = ('No\nscratchpad', '# before\ninput', '# after\ninput')
INK = '#1A1A1A'
GRID = '#E3E3E3'
SEEN_FILL = '#E6E6E6'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE / 'results_step6_nope.csv')
    parser.add_argument('--predictions', type=Path, default=HERE / 'predictions_step6_nope.csv')
    parser.add_argument('--baseline-results', type=Path, default=HERE / 'results_step5_nope.csv')
    parser.add_argument('--baseline-predictions', type=Path, default=HERE / 'predictions_step5_nope.csv')
    parser.add_argument('--data-dir', type=Path, default=HERE / 'data' / 'count_scratchpad_L40_seen10_step5')
    parser.add_argument('--run-prefix', type=Path, default=HERE / 'out_step6_nope')
    parser.add_argument('--baseline-run-prefix', type=Path, default=HERE / 'out_step5_nope')
    parser.add_argument('--summary', type=Path, default=HERE / 'results_step6_summary.csv')
    parser.add_argument('--paired', type=Path, default=HERE / 'results_step6_paired.csv')
    parser.add_argument('--figure-dir', type=Path, default=HERE / 'log' / 'figures')
    parser.add_argument('--skip-checkpoint-checks', action='store_true',
                        help='validate CSV/data metadata only, for an exported result bundle')
    parser.add_argument('--no-plots', action='store_true')
    return parser.parse_args()


def load_results(path, conditions, allow_other_conditions=False):
    indexed = {}
    with path.open(newline='') as file:
        for row in csv.DictReader(file):
            if row['condition'] not in conditions:
                if allow_other_conditions:
                    continue
                raise ValueError(f'{path}: unexpected condition {row["condition"]!r}')
            key = (row['condition'], int(row['seed']), int(row['iteration']))
            if key in indexed:
                raise ValueError(f'duplicate result in {path}: {key}')
            expected = {'pos_type': 'none', **{k: str(v) for k, v in META.items()
                                              if k != 'vocab_size'}}
            for field, value in expected.items():
                if row[field] != value:
                    raise ValueError(f'{path}: {key} has {field}={row[field]!r}, expected {value!r}')
            indexed[key] = row
    expected_keys = {(condition, seed, iteration) for condition in conditions
                     for seed in SEEDS for iteration in ITERATIONS}
    if set(indexed) != expected_keys:
        raise ValueError(f'{path}: missing={sorted(expected_keys - set(indexed))}; '
                         f'extra={sorted(set(indexed) - expected_keys)}')
    return indexed


def load_raw_data(data_dir):
    with (data_dir / 'meta.pkl').open('rb') as file:
        meta = pickle.load(file)
    for field, value in {**META, 'counts': [0, 1, 2, 3]}.items():
        if meta[field] != value:
            raise ValueError(f'dataset has {field}={meta[field]!r}, expected {value!r}')
    raw = {}
    for split, expected_size in SPLIT_SIZES.items():
        ids = np.fromfile(data_dir / f'{split}.bin', dtype=np.uint16)
        text = ''.join(meta['itos'][int(token_id)] for token_id in ids)
        examples = [tuple(line.split(':')) for line in text.splitlines() if line]
        if len(examples) != expected_size:
            raise ValueError(f'{split}: expected {expected_size} examples, found {len(examples)}')
        if len({body for body, _ in examples}) != expected_size:
            raise ValueError(f'{split}: duplicate raw inputs')
        balance = Counter(label for _, label in examples)
        if balance != Counter({str(k): expected_size // 4 for k in range(4)}):
            raise ValueError(f'{split}: unexpected class balance {balance}')
        for body, label in examples:
            positions = [i for i, char in enumerate(body) if char == 'X']
            allowed = range(10) if split == 'val' else range(10, 40)
            if len(body) != 40 or label != str(len(positions)) or not set(positions) <= set(allowed):
                raise ValueError(f'{split}: invalid example {body}:{label}')
        raw[split] = examples
    if {body for body, _ in raw['val']} & {body for body, _ in raw['test']}:
        raise ValueError('validation and test pools overlap')
    return raw


def load_predictions(path, conditions, raw, allow_other_conditions=False):
    """Check every final prediction against the same ordered raw examples."""
    counts = defaultdict(Counter)
    positions = defaultdict(lambda: [0, 0])
    with path.open(newline='') as file:
        for row in csv.DictReader(file):
            condition = row['condition']
            if condition not in conditions:
                if allow_other_conditions:
                    continue
                raise ValueError(f'{path}: unexpected condition {condition!r}')
            seed, split = int(row['seed']), row['split']
            if seed not in SEEDS or split not in raw or row['iteration'] != '2000':
                raise ValueError(f'{path}: unexpected prediction group {condition}, {seed}, {split}, {row["iteration"]}')
            if row['pos_type'] != 'none' or row['data_seed'] != '1337':
                raise ValueError(f'{path}: inconsistent prediction metadata')
            key = (condition, seed, split)
            stats = counts[key]
            index = stats['n']
            if index >= len(raw[split]):
                raise ValueError(f'{path}: extra or duplicated predictions for {key}')
            body, gold = raw[split][index]
            if (row['body'], row['gold']) != (body, gold):
                raise ValueError(f'{path}: {key} input {index} does not match the reference pool')
            expected_positions = ' '.join(str(i) for i, char in enumerate(body) if char == 'X')
            if row['x_positions'] != expected_positions:
                raise ValueError(f'{path}: incorrect stored X positions for {key}, input {index}')
            correct = row['pred'] == gold
            if row['correct'] != str(int(correct)):
                raise ValueError(f'{path}: stored correctness disagrees with the answer for {key}')
            if row['pred_states']:
                raise ValueError(f'{path}: answer-only condition unexpectedly has generated states')
            stats['n'] += 1
            stats['correct'] += correct
            stats[f'n_k{gold}'] += 1
            stats[f'correct_k{gold}'] += correct
            if row['pred'] not in {'0', '1', '2', '3'}:
                stats['invalid'] += 1
            elif int(row['pred']) < int(gold):
                stats['under'] += 1
            elif int(row['pred']) > int(gold):
                stats['over'] += 1
            if gold == '1':
                cell = positions[(condition, seed, int(expected_positions))]
                cell[0] += correct
                cell[1] += 1
    expected_keys = {(condition, seed, split) for condition in conditions
                     for seed in SEEDS for split in raw}
    if set(counts) != expected_keys:
        raise ValueError(f'{path}: incomplete prediction grid: {sorted(expected_keys - set(counts))}')
    for key, stats in counts.items():
        if stats['n'] != len(raw[key[2]]):
            raise ValueError(f'{path}: {key} has {stats["n"]} predictions')
    return counts, positions


def check_prediction_accuracies(indexed, counts):
    for (condition, seed, split), stats in counts.items():
        row = indexed[(condition, seed, 2000)]
        prefix = 'val' if split == 'val' else 'heldout'
        metrics = {
            'acc_all': stats['correct'] / stats['n'],
            'acc_k1plus': sum(stats[f'correct_k{k}'] for k in (1, 2, 3))
                           / sum(stats[f'n_k{k}'] for k in (1, 2, 3)),
            'under_rate': stats['under'] / stats['n'],
            'over_rate': stats['over'] / stats['n'],
            'invalid_rate': stats['invalid'] / stats['n'],
            **{f'acc_k{k}': stats[f'correct_k{k}'] / stats[f'n_k{k}'] for k in range(4)},
        }
        for metric, expected in metrics.items():
            actual = float(row[f'{prefix}_{metric}'])
            if not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f'{condition}, seed {seed}, {split}: {metric} '
                                 f'disagrees between results ({actual}) and predictions ({expected})')


def check_checkpoints(args):
    """Check the saved architecture and raw-data metadata for all 15 models."""
    import torch

    expected_model = {
        'vocab_size': 43, 'block_size': 96, 'n_layer': 4, 'n_head': 4,
        'n_embd': 128, 'dropout': 0.0, 'bias': True, 'pos_type': 'none',
        'causal': True, 't5_bias_mode': 'auto',
    }
    for condition in CONDITIONS:
        prefix = args.baseline_run_prefix if condition == 'no_scratchpad' else args.run_prefix
        for seed in SEEDS:
            path = Path(f'{prefix}_{condition}_{seed}') / 'ckpt_iter2000.pt'
            checkpoint = torch.load(path, map_location='cpu', weights_only=False)
            if checkpoint['model_args'] != expected_model:
                raise ValueError(f'{path}: model configuration differs: {checkpoint["model_args"]}')
            for field, expected in {**META, 'counts': [0, 1, 2, 3]}.items():
                if checkpoint['data_meta'][field] != expected:
                    raise ValueError(f'{path}: different {field} in checkpoint data metadata')
            if (checkpoint['condition'], checkpoint['seed'], checkpoint['iteration']) != (condition, seed, 2000):
                raise ValueError(f'{path}: incorrect checkpoint identity')


def metric_values(indexed, condition, iteration, metric):
    return [100.0 * float(indexed[(condition, seed, iteration)][metric]) for seed in SEEDS]


def write_summary(path, indexed):
    metrics = ('val_acc_all', 'heldout_acc_k1plus', 'heldout_acc_k0',
               'heldout_acc_k1', 'heldout_acc_k2', 'heldout_acc_k3',
               'heldout_under_rate', 'heldout_over_rate', 'heldout_invalid_rate')
    fields = ['condition', 'iteration', 'n_seeds', 'raw_length', 'model_input_length']
    fields += [f'{metric}_{suffix}_pct' for metric in metrics for suffix in ('mean', 'sample_sd')]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for condition in CONDITIONS:
            for iteration in ITERATIONS:
                row = {'condition': condition, 'iteration': iteration, 'n_seeds': 5,
                       'raw_length': 40, 'model_input_length': 41 if condition == 'no_scratchpad' else 42}
                for metric in metrics:
                    values = metric_values(indexed, condition, iteration, metric)
                    row[f'{metric}_mean_pct'] = f'{mean(values):.4f}'
                    row[f'{metric}_sample_sd_pct'] = f'{stdev(values):.4f}'
                writer.writerow(row)


def write_paired(path, indexed):
    comparisons = (('prefix_one', 'no_scratchpad'), ('suffix_one', 'no_scratchpad'),
                   ('prefix_one', 'suffix_one'))
    fields = ['condition', 'reference', 'seed', 'condition_acc_k1plus_pct',
              'reference_acc_k1plus_pct', 'difference_k1plus_pp']
    fields += [f'difference_k{k}_pp' for k in range(4)]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for condition, reference in comparisons:
            differences = []
            for seed in SEEDS:
                current = indexed[(condition, seed, 2000)]
                baseline = indexed[(reference, seed, 2000)]
                value = 100 * float(current['heldout_acc_k1plus'])
                base = 100 * float(baseline['heldout_acc_k1plus'])
                differences.append(value - base)
                row = {'condition': condition, 'reference': reference, 'seed': seed,
                       'condition_acc_k1plus_pct': f'{value:.4f}',
                       'reference_acc_k1plus_pct': f'{base:.4f}',
                       'difference_k1plus_pp': f'{value - base:.4f}'}
                for count in range(4):
                    metric = f'heldout_acc_k{count}'
                    row[f'difference_k{count}_pp'] = f'{100 * (float(current[metric]) - float(baseline[metric])):.4f}'
                writer.writerow(row)
            print(f'{condition} - {reference}: mean {mean(differences):+.2f} pp; '
                  f'positive in {sum(value > 0 for value in differences)}/5 seeds; '
                  f'paired differences {[round(value, 2) for value in differences]}')


def configure_plotting():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.labelsize': 9,
        'axes.titlesize': 10, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'axes.edgecolor': INK, 'axes.labelcolor': INK, 'text.color': INK,
        'xtick.color': INK, 'ytick.color': INK, 'axes.linewidth': 0.7,
        'xtick.major.width': 0.7, 'ytick.major.width': 0.7,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
    })


def style_axis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_yticks([0, 25, 50, 75, 100])


def save_figure(figure, directory, stem):
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        path = directory / f'{stem}.{extension}'
        figure.savefig(path, facecolor='white', bbox_inches='tight')
        print(f'wrote {path}')
    plt.close(figure)


def plot_accuracy(indexed, directory):
    import matplotlib.pyplot as plt

    panels = (('heldout_acc_k1plus', '(a) Counts 1–3'),
              ('heldout_acc_k1', '(b) One X'), ('heldout_acc_k2', '(c) Two X'),
              ('heldout_acc_k3', '(d) Three X'))
    figure, axes = plt.subplots(2, 2, figsize=(7.5, 5.4), sharey=True)
    jitter = np.linspace(-0.13, 0.13, len(SEEDS))
    for ax, (metric, title) in zip(axes.flat, panels):
        for index, (condition, color, marker) in enumerate(zip(CONDITIONS, COLORS, MARKERS)):
            values = metric_values(indexed, condition, 2000, metric)
            ax.scatter(index + jitter, values, s=27, marker=marker, color=color,
                       edgecolor=INK, linewidth=0.45, zorder=3)
            average = mean(values)
            ax.plot([index - 0.19, index + 0.19], [average, average],
                    color=INK, linewidth=1.8, zorder=4)
            ax.text(index, 108, f'{average:.1f} ± {stdev(values):.1f}',
                    ha='center', va='center', fontsize=8)
        style_axis(ax)
        ax.set_ylim(-4, 115)
        ax.set_xlim(-0.48, 2.48)
        ax.set_xticks(range(3), TICK_LABELS)
        ax.set_title(title, loc='left', pad=6)
        ax.set_ylabel('Unseen-position accuracy (%)')
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.10, top=0.95,
                           wspace=0.25, hspace=0.43)
    save_figure(figure, directory, 'step6_single_token_accuracy')


def plot_main_metric(indexed, directory):
    """One primary outcome, with paired seeds visible across all three formats."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, ax = plt.subplots(figsize=(6.0, 4.0))
    figure.text(0.55, 0.975, 'Unseen X positions · counts 1–3',
                ha='center', va='top', fontsize=10)
    jitter = np.linspace(-0.10, 0.10, len(SEEDS))
    scores = np.asarray([
        metric_values(indexed, condition, 2000, 'heldout_acc_k1plus')
        for condition in CONDITIONS
    ])
    for seed_index, offset in enumerate(jitter):
        ax.plot(np.arange(3) + offset, scores[:, seed_index], color='#B6B6B6',
                linewidth=0.8, zorder=2)
    for index, (color, marker) in enumerate(zip(COLORS, MARKERS)):
        ax.scatter(index + jitter, scores[index], s=34, marker=marker,
                   color=color, edgecolor=INK, linewidth=0.5, zorder=3)
        average = mean(scores[index])
        ax.plot([index - 0.17, index + 0.17], [average, average],
                color=INK, linewidth=2.0, solid_capstyle='butt', zorder=2.5)
        ax.text(index, 1.035, f'Mean {average:.1f}%',
                transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                fontsize=10, clip_on=False)
    style_axis(ax)
    ax.set_xlim(-0.43, 2.43)
    ax.set_ylim(0, 103)
    ax.set_xticks(range(3), ('No extra token\nbody:count',
                           '# before input\n#body:count',
                           '# after input\nbody#:count'))
    ax.tick_params(axis='both', labelsize=9)
    ax.tick_params(axis='x', pad=7)
    ax.set_ylabel('Final-answer accuracy (%)', fontsize=10)
    handles = [
        Line2D([], [], color='#B6B6B6', marker='o', markersize=4,
               linewidth=0.8, label='Paired model seeds (n = 5)'),
        Line2D([], [], color=INK, linewidth=2.0, label='Mean'),
    ]
    ax.legend(handles=handles, loc='lower left', frameon=False, fontsize=8.5,
              handlelength=2.0, labelspacing=0.55, borderaxespad=0.8)
    figure.subplots_adjust(left=0.12, right=0.98, bottom=0.19, top=0.84)
    save_figure(figure, directory, 'step6_single_token_main')


def plot_positions(positions, directory):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 3, figsize=(7.8, 2.65), sharey=True)
    x = np.arange(40)
    for ax, condition, label, color, marker in zip(axes, CONDITIONS, LABELS, COLORS, MARKERS):
        curves = []
        for seed in SEEDS:
            values = []
            for position in x:
                correct, total = positions[(condition, seed, int(position))]
                if not total:
                    raise ValueError(f'missing single-X coverage: {condition}, seed {seed}, position {position}')
                values.append(100 * correct / total)
            curves.append(values)
        curves = np.asarray(curves)
        ax.axvspan(-0.5, 9.5, color=SEEN_FILL, linewidth=0)
        ax.axvline(9.5, color='#9B9B9B', linewidth=0.7)
        ax.fill_between(x, curves.min(axis=0), curves.max(axis=0),
                        color=color, alpha=0.17, linewidth=0)
        ax.plot(x, curves.mean(axis=0), color=color, linewidth=1.4,
                marker=marker, markersize=3.1, markevery=5,
                markeredgecolor='white', markeredgewidth=0.35)
        style_axis(ax)
        ax.set_xlim(-0.5, 39.5)
        ax.set_ylim(-4, 111)
        ax.set_xticks([0, 10, 20, 30, 39])
        ax.set_title(label, pad=10)
        ax.text(4.5, 105, 'Seen', ha='center', fontsize=7.5, color='#666666')
        ax.text(24.5, 105, 'Unseen', ha='center', fontsize=7.5, color='#666666')
        ax.set_xlabel('Position of the single X')
    axes[0].set_ylabel('Accuracy (%)')
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.84, wspace=0.20)
    save_figure(figure, directory, 'step6_single_token_positions')


def main():
    args = parse_args()
    indexed = load_results(args.baseline_results, ('no_scratchpad',), allow_other_conditions=True)
    indexed.update(load_results(args.results, NEW_CONDITIONS))
    raw = load_raw_data(args.data_dir)
    counts, positions = load_predictions(args.baseline_predictions, ('no_scratchpad',), raw,
                                         allow_other_conditions=True)
    new_counts, new_positions = load_predictions(args.predictions, NEW_CONDITIONS, raw)
    counts.update(new_counts)
    positions.update(new_positions)
    check_prediction_accuracies(indexed, counts)
    if not args.skip_checkpoint_checks:
        check_checkpoints(args)
    print('Validated 60 result rows and 105,000 final predictions for 15 models.')
    write_summary(args.summary, indexed)
    write_paired(args.paired, indexed)
    for condition in CONDITIONS:
        values = metric_values(indexed, condition, 2000, 'heldout_acc_k1plus')
        per_count = [mean(metric_values(indexed, condition, 2000, f'heldout_acc_k{k}'))
                     for k in range(4)]
        print(f'{condition}: unseen counts 1–3 = {mean(values):.2f} ± {stdev(values):.2f}%; '
              f'counts 0/1/2/3 = {[round(value, 2) for value in per_count]}')
    print(f'wrote {args.summary}')
    print(f'wrote {args.paired}')
    if not args.no_plots:
        configure_plotting()
        plot_main_metric(indexed, args.figure_dir)
        plot_accuracy(indexed, args.figure_dir)
        plot_positions(positions, args.figure_dir)


if __name__ == '__main__':
    main()
