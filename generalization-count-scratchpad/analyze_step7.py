"""Validate and summarize frozen-model running-count probes (Step 7).

This script fits no probes and changes no models. It checks the complete output
grid from probe_step7.py, writes seed summaries, and plots layer and control
comparisons. Count decodability is a representation diagnostic, not a causal test.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


HERE = Path(__file__).resolve().parent
CONDITIONS = ('no_scratchpad', 'dummy_given', 'dummy')
SITES = {'no_scratchpad': ('raw',), 'dummy_given': ('raw', 'hash'),
         'dummy': ('raw', 'hash')}
SEEDS = (1337, 1338, 1339, 1340, 1341)
MODEL_KINDS = ('trained', 'random')
LABEL_KINDS = ('true', 'shuffled')
POOLS = ('seen', 'unseen')
POOL_SIZE = {'fit': 1792, 'seen': 896, 'unseen': 3456}
POOL_POSITIONS = {'fit': tuple(range(3, 10)), 'seen': tuple(range(3, 10)),
                  'unseen': tuple(range(13, 40))}
SITE_ORDER = (('no_scratchpad', 'raw'), ('dummy_given', 'raw'),
              ('dummy_given', 'hash'), ('dummy', 'raw'), ('dummy', 'hash'))
CONDITION_COLORS = {'no_scratchpad': '#0072B2', 'dummy_given': '#56B4E9',
                    'dummy': '#D55E00'}
CONDITION_LABELS = {'no_scratchpad': 'No scratchpad', 'dummy_given': 'Dummy, given',
                    'dummy': 'Dummy, predicted'}
INK = '#1A1A1A'
GRID = '#E3E3E3'


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=HERE / 'results_step7_probes.csv')
    parser.add_argument('--positions', type=Path, default=HERE / 'results_step7_positions.csv')
    parser.add_argument('--diagnostics', type=Path, default=HERE / 'results_step7_diagnostics.csv')
    parser.add_argument('--manifest', type=Path, default=HERE / 'step7_manifest.json')
    parser.add_argument('--data-dir', type=Path, default=HERE / 'data/count_scratchpad_L40_seen10_step5')
    parser.add_argument('--summary', type=Path, default=HERE / 'results_step7_summary.csv')
    parser.add_argument('--figure-dir', type=Path, default=HERE / 'log/figures')
    parser.add_argument('--no-plots', action='store_true')
    parser.add_argument('--skip-data-hashes', action='store_true',
                        help='allow analysis of an exported bundle without the raw dataset')
    return parser.parse_args()


def expected_keys():
    return {
        (condition, seed, model_kind, site, layer, labels, pool)
        for condition in CONDITIONS
        for seed, model_kind, site, layer, labels, pool in itertools.product(
            SEEDS, MODEL_KINDS, SITES[condition], range(5), LABEL_KINDS, POOLS)
    }


def identity(row):
    return (row['condition'], int(row['seed']), row['model_kind'], row['site'],
            int(row['layer']), row['labels'], row['pool'])


def accuracy(value, location):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f'{location}: expected an accuracy in [0,1], found {value}')
    return value


def require_close(actual, expected, location, tolerance=1e-10):
    if not math.isclose(actual, expected, rel_tol=0, abs_tol=tolerance):
        raise ValueError(f'{location}: {actual} does not equal {expected}')


def require_grid(actual, expected, source):
    if set(actual) != expected:
        missing, extra = expected - set(actual), set(actual) - expected
        raise ValueError(f'{source}: incomplete grid; missing={sorted(missing)[:5]} '
                         f'({len(missing)} total), extra={sorted(extra)[:5]} ({len(extra)} total)')


def load_results(path):
    indexed = {}
    fields = ('accuracy', 'macro_recall', 'recall_k0', 'recall_k1', 'recall_k2', 'recall_k3')
    with path.open(newline='') as file:
        for row in csv.DictReader(file):
            key = identity(row)
            if key in indexed:
                raise ValueError(f'{path}: duplicate row {key}')
            if row['pool'] not in POOLS or int(row['n']) != POOL_SIZE[row['pool']]:
                raise ValueError(f'{path}: wrong pool or sample count for {key}')
            for field in fields:
                row[field] = accuracy(row[field], f'{path}, {key}, {field}')
            require_close(row['macro_recall'], mean(row[f'recall_k{k}'] for k in range(4)),
                          f'{path}, {key}, macro recall')
            require_close(row['accuracy'], row['macro_recall'],
                          f'{path}, {key}, balanced accuracy')
            if key[4] == 0:
                require_close(row['accuracy'], 0.25, f'{path}, {key}, layer-0 control')
            indexed[key] = row
    require_grid(indexed, expected_keys(), path)
    return indexed


def load_positions(path, results):
    indexed = {}
    with path.open(newline='') as file:
        for row in csv.DictReader(file):
            base_key = identity(row)
            key = base_key + (int(row['position']),)
            if key in indexed:
                raise ValueError(f'{path}: duplicate position row {key}')
            if base_key[4] != 4 or base_key not in results:
                raise ValueError(f'{path}: unknown model/site/label key or non-primary layer: {base_key}')
            if int(row['n']) != 128:
                raise ValueError(f'{path}: expected 128 examples per position, found {row["n"]}')
            row['accuracy'] = accuracy(row['accuracy'], f'{path}, {key}')
            indexed[key] = row
    expected = {key + (position,) for key in expected_keys() if key[4] == 4
                for position in POOL_POSITIONS[key[6]]}
    require_grid(indexed, expected, path)
    for key in expected_keys():
        if key[4] != 4:
            continue
        values = [indexed[key + (position,)]['accuracy'] for position in POOL_POSITIONS[key[6]]]
        require_close(mean(values), results[key]['accuracy'],
                      f'{path}, {key}, position aggregate')
    return indexed


def load_diagnostics(path):
    indexed = {}
    with path.open(newline='') as file:
        for row in csv.DictReader(file):
            key = (row['condition'], int(row['seed']), row['model_kind'], row['pool'])
            if key in indexed:
                raise ValueError(f'{path}: duplicate diagnostic row {key}')
            if row['pool'] not in POOL_SIZE or int(row['n']) != POOL_SIZE[row['pool']]:
                raise ValueError(f'{path}: wrong diagnostic sample count for {key}')
            row['answer_acc'] = accuracy(row['answer_acc'], f'{path}, {key}, answer')
            hash_accuracy = row['query_hash_prediction_acc']
            if row['condition'] == 'dummy':
                row['query_hash_prediction_acc'] = accuracy(hash_accuracy, f'{path}, {key}, hash')
            elif hash_accuracy != '':
                raise ValueError(f'{path}: hash-prediction diagnostic should be empty for {key}')
            indexed[key] = row
    expected = set(itertools.product(CONDITIONS, SEEDS, MODEL_KINDS, POOL_SIZE))
    require_grid(indexed, expected, path)
    return indexed


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_manifest(args):
    with args.manifest.open() as file:
        manifest = json.load(file)
    if (manifest['pool_seed'] != 20261108 or manifest['primary_layer'] != 4
            or tuple(manifest['model_seeds']) != SEEDS
            or tuple(manifest['conditions']) != CONDITIONS):
        raise ValueError('manifest differs from the pre-set Step 7 design')
    if manifest['probe'] != 'four-class ridge; fit-only standardization; unnormalized SSE + L2(1.0); free intercept':
        raise ValueError('manifest probe objective differs from the pre-set design')
    if set(manifest['pools']) != set(POOL_SIZE):
        raise ValueError('manifest has unexpected pools')
    for pool, n in POOL_SIZE.items():
        entry = manifest['pools'][pool]
        if (entry['n'] != n or entry['positions'] != list(POOL_POSITIONS[pool])
                or entry['labels'] != {str(k): n // 4 for k in range(4)}):
            raise ValueError(f'manifest pool differs from balanced design: {pool}')
        if len(entry['sha256']) != 64 or any(char not in '0123456789abcdef' for char in entry['sha256']):
            raise ValueError(f'missing pool checksum: {pool}')
    if len(manifest['checkpoint_sha256']) != 15:
        raise ValueError('expected 15 checkpoint checksums')
    for path, digest in manifest['checkpoint_sha256'].items():
        if len(digest) != 64 or any(char not in '0123456789abcdef' for char in digest):
            raise ValueError(f'invalid checkpoint checksum: {path}')
    if not args.skip_data_hashes:
        for name in ('train.bin', 'val.bin', 'test.bin', 'meta.pkl'):
            if checksum(args.data_dir / name) != manifest['data_sha256'][name]:
                raise ValueError(f'dataset hash differs from extraction manifest: {name}')
    return manifest


def values(results, condition, site, layer, pool, model_kind='trained', labels='true', metric='accuracy'):
    return [100 * results[(condition, seed, model_kind, site, layer, labels, pool)][metric]
            for seed in SEEDS]


def write_summary(path, results):
    groups = defaultdict(list)
    for key, row in results.items():
        condition, _seed, model_kind, site, layer, labels, pool = key
        groups[(condition, site, layer, model_kind, labels, pool)].append(row)
    metrics = ('accuracy', 'macro_recall', 'recall_k0', 'recall_k1', 'recall_k2', 'recall_k3')
    fields = ['condition', 'site', 'layer', 'model_kind', 'labels', 'pool', 'n_seeds', 'n_per_seed']
    fields += [f'{metric}_{stat}_pct' for metric in metrics for stat in ('mean', 'sample_sd')]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        for group, rows in sorted(groups.items()):
            if len(rows) != 5:
                raise ValueError(f'incomplete seed group {group}')
            row = dict(zip(fields[:6], group))
            row.update(n_seeds=5, n_per_seed=int(rows[0]['n']))
            for metric in metrics:
                scores = [100 * item[metric] for item in rows]
                row[f'{metric}_mean_pct'] = f'{mean(scores):.6f}'
                row[f'{metric}_sample_sd_pct'] = f'{stdev(scores):.6f}'
            writer.writerow(row)


def configure_plotting():
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 8.5, 'axes.labelsize': 9,
        'axes.titlesize': 9.5, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
        'legend.fontsize': 7.5, 'axes.edgecolor': INK, 'axes.labelcolor': INK,
        'text.color': INK, 'xtick.color': INK, 'ytick.color': INK,
        'axes.linewidth': 0.7, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 300,
    })


def style_axis(ax):
    for spine in ('top', 'right'):
        ax.spines[spine].set_visible(False)
    ax.grid(axis='y', color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.axhline(25, color='#666666', linestyle=(0, (4, 3)), linewidth=0.8)
    ax.tick_params(length=3, width=0.7)


def save_figure(figure, directory, stem):
    import matplotlib.pyplot as plt

    directory.mkdir(parents=True, exist_ok=True)
    for extension in ('png', 'pdf'):
        path = directory / f'{stem}.{extension}'
        figure.savefig(path, facecolor='white', bbox_inches='tight')
        print(f'wrote {path}')
    plt.close(figure)


def plot_layers(results, directory):
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(7.5, 3.4), sharey=True)
    for ax, pool, title in zip(axes, POOLS, ('(a) Seen positions', '(b) Unseen positions')):
        for condition, site in SITE_ORDER:
            by_layer = [values(results, condition, site, layer, pool) for layer in range(5)]
            average = [mean(scores) for scores in by_layer]
            # A seed-range band stays inside 0–100%; summary CSV reports sample SD.
            ax.fill_between(range(5), [min(scores) for scores in by_layer],
                            [max(scores) for scores in by_layer],
                            color=CONDITION_COLORS[condition], alpha=0.09, linewidth=0)
            ax.plot(range(5), average, color=CONDITION_COLORS[condition],
                    marker='o' if site == 'raw' else 's', markersize=4,
                    markerfacecolor='white' if site == 'raw' else CONDITION_COLORS[condition],
                    markeredgewidth=0.8, linewidth=1.25,
                    linestyle='-' if site == 'raw' else '--',
                    label=f'{CONDITION_LABELS[condition]} — {"raw" if site == "raw" else "#"}')
        style_axis(ax)
        ax.set_xlim(-0.18, 4.18)
        ax.set_xticks(range(5))
        ax.set_xlabel('Residual-stream layer (0 = embedding)')
        ax.set_title(title, loc='left')
    axes[0].set_ylabel('Linear-probe accuracy (%)')
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.52, 1.02),
                  ncol=2, frameon=False, columnspacing=1.7, handlelength=2.3)
    figure.subplots_adjust(left=0.085, right=0.99, bottom=0.17, top=0.71, wspace=0.20)
    save_figure(figure, directory, 'step7_probe_layers')


def plot_primary_controls(results, directory):
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(7.5, 3.45))
    controls = (('trained', 'true', -0.23, 'o', 'Trained model / true labels'),
                ('random', 'true', 0.0, '^', 'Random model / true labels'),
                ('trained', 'shuffled', 0.23, 's', 'Trained model / shuffled fit labels'))
    jitter = (-0.055, -0.0275, 0, 0.0275, 0.055)
    for index, (condition, site) in enumerate(SITE_ORDER):
        for model_kind, labels, offset, marker, control_label in controls:
            scores = values(results, condition, site, 4, 'unseen', model_kind, labels)
            color = CONDITION_COLORS[condition] if model_kind == 'trained' and labels == 'true' else '#777777'
            ax.scatter([index + offset + delta for delta in jitter], scores,
                       color=color if labels == 'true' else 'white', edgecolor=color,
                       marker=marker, s=21, linewidth=0.8, zorder=3,
                       label=control_label if index == 0 else None)
            ax.plot([index + offset - 0.085, index + offset + 0.085],
                    [mean(scores), mean(scores)], color=INK, linewidth=1.5, zorder=4)
    style_axis(ax)
    ax.set_xlim(-0.5, 4.5)
    ax.set_xticks(range(5), ('No scratchpad\nraw', 'Dummy, given\nraw',
                            'Dummy, given\n#', 'Dummy, predicted\nraw', 'Dummy, predicted\n#'))
    ax.set_ylabel('Unseen-position probe accuracy (%)')
    ax.set_xlabel('Measurement site at residual-stream layer 4')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.27), frameon=False,
              ncol=1, handletextpad=0.5)
    figure.subplots_adjust(left=0.085, right=0.99, bottom=0.23, top=0.75)
    save_figure(figure, directory, 'step7_probe_controls')


def plot_positions(positions, directory):
    """Layer-4 trained/true-label readouts by raw query position.

    Seen and unseen queries are separate pools. Drawing them in separate calls
    leaves the unmeasured positions 10–12 unconnected rather than interpolating.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    figure, axes = plt.subplots(1, 3, figsize=(7.8, 2.85), sharey=True)
    for ax, condition in zip(axes, CONDITIONS):
        color = CONDITION_COLORS[condition]
        ax.axvspan(2.5, 9.5, color='#E6E6E6', linewidth=0, zorder=0)
        for site in SITES[condition]:
            for pool in POOLS:
                queries = POOL_POSITIONS[pool]
                columns = [
                    [100 * positions[(condition, seed, 'trained', site, 4,
                                      'true', pool, position)]['accuracy']
                     for seed in SEEDS]
                    for position in queries
                ]
                ax.fill_between(queries, [min(scores) for scores in columns],
                                [max(scores) for scores in columns], color=color,
                                alpha=0.13, linewidth=0, zorder=1)
                ax.plot(queries, [mean(scores) for scores in columns],
                        color=color, linestyle='-' if site == 'raw' else '--',
                        marker='o' if site == 'raw' else 's', markersize=3.3,
                        markevery=2 if pool == 'seen' else 4,
                        markerfacecolor='white' if site == 'raw' else color,
                        markeredgewidth=0.65, linewidth=1.3, zorder=3)
        style_axis(ax)
        ax.set_xlim(-0.5, 39.5)
        ax.set_ylim(0, 110)
        ax.set_xticks([0, 10, 20, 30, 39])
        ax.set_xlabel('Raw query position (0–39)')
        ax.set_title(CONDITION_LABELS[condition], pad=10)
        ax.text(6, 105, 'Seen', ha='center', va='center', color='#666666', fontsize=7.5)
        ax.text(26, 105, 'Unseen', ha='center', va='center', color='#666666', fontsize=7.5)
    axes[0].set_ylabel('Linear-probe accuracy (%)')
    handles = [
        Line2D([], [], color=INK, marker='o', markerfacecolor='white',
               markersize=4, linewidth=1.3, label='Current raw character'),
        Line2D([], [], color=INK, linestyle='--', marker='s', markersize=4,
               linewidth=1.3, label='Following #'),
    ]
    figure.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.53, 1.025),
                  ncol=2, frameon=False, handlelength=2.6, columnspacing=2)
    figure.subplots_adjust(left=0.075, right=0.99, bottom=0.19, top=0.77, wspace=0.20)
    save_figure(figure, directory, 'step7_probe_positions')


def plot_hidden_count_main(results, directory):
    """Representative figure: one fixed # site, with its controls and limits.

    The same fitted four-class readouts are used in both panels. This figure
    changes presentation only; the full site/control/layer results remain in the
    other figures and summary. Pairing connects equal model seeds within a pool
    or count, and every seed is shown rather than an uncertainty interval.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    controls = (
        ('trained', -0.17, 'o', '#D55E00', 'Trained model'),
        ('random', 0.17, '^', '#777777', 'Untrained model'),
    )
    jitter = (-0.065, -0.0325, 0, 0.0325, 0.065)
    with plt.rc_context({'font.size': 9.5, 'axes.labelsize': 9.5,
                         'axes.titlesize': 10, 'xtick.labelsize': 9,
                         'ytick.labelsize': 9, 'legend.fontsize': 9.5}):
        figure, axes = plt.subplots(1, 2, figsize=(7.9, 3.65), sharey=True)
        for ax in axes:
            for spine in ('top', 'right'):
                ax.spines[spine].set_visible(False)
            ax.set_axisbelow(True)
            ax.grid(axis='y', color=GRID, linewidth=0.6)
            ax.set_ylim(-2, 110)
            ax.set_yticks([0, 25, 50, 75, 100])
            ax.tick_params(length=3, width=0.7)

        def draw_pair(ax, index, trained, untrained, annotate=False):
            for seed_index, (left, right) in enumerate(zip(trained, untrained)):
                delta = jitter[seed_index]
                ax.plot([index - 0.17 + delta, index + 0.17 + delta],
                        [left, right], color='#C6C6C6', linewidth=0.7,
                        alpha=0.8, zorder=1)
            for (kind, offset, marker, color, _label), scores in zip(
                    controls, (trained, untrained)):
                ax.scatter([index + offset + delta for delta in jitter], scores,
                           s=25, marker=marker, color=color, edgecolor='white',
                           linewidth=0.45, zorder=3)
                ax.plot([index + offset - 0.105, index + offset + 0.105],
                        [mean(scores), mean(scores)], color=INK, linewidth=1.8,
                        solid_capstyle='butt', zorder=2)
                if annotate:
                    ax.text(index + offset, 106, f'{mean(scores):.2f}',
                            ha='center', va='center', fontsize=9.5, color=INK)

        for index, pool in enumerate(POOLS):
            trained = values(results, 'dummy', 'hash', 4, pool, 'trained')
            untrained = values(results, 'dummy', 'hash', 4, pool, 'random')
            draw_pair(axes[0], index, trained, untrained, annotate=True)
        axes[0].axhline(25, color='#666666', linestyle=(0, (4, 3)), linewidth=0.8)
        axes[0].text(1.45, 27.5, 'Chance (25%)', color='#555555', fontsize=8.5,
                     ha='right', va='bottom')
        axes[0].set_xlim(-0.48, 1.48)
        axes[0].set_xticks((0, 1), ('Seen\n(3–9)', 'Unseen\n(13–39)'))
        axes[0].set_xlabel('Query positions')
        axes[0].set_ylabel('Count decoding accuracy (%)')
        axes[0].set_title('(a) Transfer to unseen positions', loc='left', pad=9)

        for count in range(4):
            metric = f'recall_k{count}'
            trained = values(results, 'dummy', 'hash', 4, 'unseen', 'trained', metric=metric)
            untrained = values(results, 'dummy', 'hash', 4, 'unseen', 'random', metric=metric)
            draw_pair(axes[1], count, trained, untrained)
        axes[1].set_xlim(-0.48, 3.48)
        axes[1].set_xticks(range(4))
        axes[1].set_xlabel('Number of Xs seen so far')
        axes[1].set_ylabel('Correctly decoded within count (%)')
        axes[1].tick_params(labelleft=True)
        axes[1].set_title('(b) Which counts are recovered?', loc='left', pad=9)

        handles = [Line2D([], [], linestyle='none', marker=marker, color=color,
                          markersize=5, label=label)
                   for _kind, _offset, marker, color, label in controls]
        figure.legend(handles=handles, loc='upper center', ncol=2, frameon=False,
                      bbox_to_anchor=(0.53, 1.015), columnspacing=2.5,
                      handletextpad=0.45)
        figure.subplots_adjust(left=0.075, right=0.99, bottom=0.20, top=0.80,
                               wspace=0.44)
        save_figure(figure, directory, 'step7_hidden_count_main')


def print_summary(results, diagnostics):
    print('Primary layer 4 linear-probe accuracy (%; mean ± sample SD across model seeds):')
    for condition, site in SITE_ORDER:
        summaries = []
        for pool in POOLS:
            trained = values(results, condition, site, 4, pool)
            random = values(results, condition, site, 4, pool, 'random')
            shuffled = values(results, condition, site, 4, pool, 'trained', 'shuffled')
            summaries.append(f'{pool}: trained {mean(trained):.2f} ± {stdev(trained):.2f}; '
                             f'random {mean(random):.2f}; shuffled {mean(shuffled):.2f}')
        print(f'  {condition}, {site}: ' + ' | '.join(summaries))
    print('Supplied-format final-answer accuracy on diagnostic pools (not original task scores):')
    for condition in CONDITIONS:
        text = []
        for pool in POOLS:
            scores = [100 * diagnostics[(condition, seed, 'trained', pool)]['answer_acc'] for seed in SEEDS]
            text.append(f'{pool} {mean(scores):.2f} ± {stdev(scores):.2f}')
        print(f'  {condition}: ' + ' | '.join(text))
    for pool in ('fit', 'seen', 'unseen'):
        scores = [100 * diagnostics[('dummy', seed, 'trained', pool)]['query_hash_prediction_acc']
                  for seed in SEEDS]
        print(f'  trained dummy query-# prediction, {pool}: {mean(scores):.2f} ± {stdev(scores):.2f}%')


def main():
    args = parse_args()
    validate_manifest(args)
    results = load_results(args.results)
    positions = load_positions(args.positions, results)
    diagnostics = load_diagnostics(args.diagnostics)
    print(f'Validated {len(results)} probe rows, {len(positions)} position rows, '
          f'{len(diagnostics)} diagnostic rows; all layer-0 controls are 25%.')
    write_summary(args.summary, results)
    print(f'wrote {args.summary}')
    print_summary(results, diagnostics)
    if not args.no_plots:
        configure_plotting()
        plot_layers(results, args.figure_dir)
        plot_primary_controls(results, args.figure_dir)
        plot_positions(positions, args.figure_dir)
        plot_hidden_count_main(results, args.figure_dir)


if __name__ == '__main__':
    main()
