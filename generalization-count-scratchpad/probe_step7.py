"""Frozen-model running-count probes with balanced, independent position pools.

Run from this folder: ../venv/bin/python probe_step7.py
Only small linear readouts are fitted; transformer weights stay frozen.
"""
import argparse
import csv
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from dataset import load_meta, load_raw_examples, tensorize_examples
from engine import MicroTransformer, MicroTransformerConfig


HERE = Path(__file__).resolve().parent
CONDITIONS = ('no_scratchpad', 'dummy_given', 'dummy')
SEEDS = (1337, 1338, 1339, 1340, 1341)
POOL_SEED = 20261108


def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def make_pool(meta, name, positions, per_cell, forbidden):
    rng = random.Random(f'{POOL_SEED}:{name}')
    region_start = 10 if name == 'unseen' else 0
    examples, queries = [], []
    for position in positions:
        if position - region_start < 3:
            raise ValueError('query must allow all four prefix counts')
        for _ in range(per_cell):
            while True:
                template = [rng.choice(meta['filler']) for _ in range(40)]
                candidates = rng.sample(list(range(region_start, position)), 3)
                quartet = []
                for count in range(4):
                    chars = template.copy()
                    for index in candidates[:count]:
                        chars[index] = 'X'
                    quartet.append((''.join(chars), str(count)))
                if all(body not in forbidden for body, _ in quartet):
                    break
            for body, label in quartet:
                forbidden.add(body)
                examples.append((body, label))
                queries.append(position)
    return examples, np.asarray(queries, dtype=np.int64)


def build_pools(meta, data_dir, train_cells=64, eval_cells=32):
    forbidden = {
        body for name in ('train', 'val', 'test')
        for body, _ in load_raw_examples(str(data_dir), name, meta)
    }
    pools = {}
    for name, positions, cells in (
        ('fit', range(3, 10), train_cells),
        ('seen', range(3, 10), eval_cells),
        ('unseen', range(13, 40), eval_cells),
    ):
        pools[name] = make_pool(meta, name, positions, cells, forbidden)
    return pools


def fit_ridge(features, labels, alpha=1.0):
    """Four-class ridge regression, centered fit-only scaling, free intercept."""
    x = features.astype(np.float64)
    center, scale = x.mean(axis=0), x.std(axis=0)
    scale[scale < 1e-8] = 1.0
    x = (x - center) / scale
    targets = np.eye(4, dtype=np.float64)[labels]
    intercept = targets.mean(axis=0)
    # Use PyTorch float64 for the small solve. The macOS NumPy BLAS backend can
    # report spurious floating-point flags on these otherwise finite products.
    tx = torch.from_numpy(x)
    ty = torch.from_numpy(targets - intercept)
    gram = tx.T @ tx + alpha * torch.eye(x.shape[1], dtype=torch.float64)
    weights = torch.linalg.solve(gram, tx.T @ ty).numpy()
    if not all(np.isfinite(part).all() for part in (center, scale, weights, intercept)):
        raise FloatingPointError('Non-finite ridge parameters')
    return center, scale, weights, intercept


def ridge_predict(features, parameters):
    center, scale, weights, intercept = parameters
    x = torch.from_numpy((features - center) / scale)
    scores = x @ torch.from_numpy(weights) + torch.from_numpy(intercept)
    if not torch.isfinite(scores).all():
        raise FloatingPointError('Non-finite ridge scores')
    return scores.argmax(dim=1).numpy()


def site_indices(condition, queries):
    if condition == 'no_scratchpad':
        return {'raw': queries}
    return {'raw': 2 * queries + 1, 'hash': 2 * queries + 2}


@torch.no_grad()
def extract(model, inputs, labels, queries, condition, batch_size, hash_id):
    """Hooks observe normal forward passes; no manual reimplementation of layers."""
    sites = site_indices(condition, queries)
    chunks = {(site, layer): [] for site in sites for layer in range(5)}
    active = {}
    hooks = []

    def capture(layer):
        def hook(_module, _inputs, output):
            rows = torch.arange(output.shape[0])
            for site in sites:
                indices = torch.from_numpy(sites[site][active['start']:active['end']])
                chunks[(site, layer)].append(output[rows, indices].cpu().numpy().copy())
        return hook

    hooks.append(model.drop.register_forward_hook(capture(0)))
    for layer, block in enumerate(model.blocks, start=1):
        hooks.append(block.register_forward_hook(capture(layer)))
    answer_hits = state_hits = total = 0
    try:
        for start in range(0, len(inputs), batch_size):
            end = min(start + batch_size, len(inputs))
            active.update(start=start, end=end)
            logits = model(inputs[start:end])
            answer_hits += int((logits[:, -1].argmax(-1).cpu().numpy() == labels[start:end]).sum())
            if condition == 'dummy':
                rows = torch.arange(end - start)
                raw_indices = torch.from_numpy(sites['raw'][start:end])
                state_hits += int((logits[rows, raw_indices].argmax(-1) == hash_id).sum())
            total += end - start
    finally:
        for hook in hooks:
            hook.remove()
    features = {key: np.concatenate(parts) for key, parts in chunks.items()}
    return features, answer_hits, state_hits, total


def feature_extract(model, inputs, examples, queries, condition, meta, batch_size):
    # Output vocabulary ids differ from class indices: score direct labels separately.
    label_ids = np.asarray([meta['stoi'][label] for _, label in examples])
    return extract(model, inputs, label_ids, queries, condition, batch_size, meta['stoi']['#'])


def write_csv(path, rows):
    with path.open('w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=HERE / 'data/count_scratchpad_L40_seen10_step5')
    parser.add_argument('--checkpoint-prefix', type=Path, default=HERE / 'out_step5_nope')
    parser.add_argument('--out-dir', type=Path, default=HERE / 'out_step7_probes')
    parser.add_argument('--results', type=Path, default=HERE / 'results_step7_probes.csv')
    parser.add_argument('--positions', type=Path, default=HERE / 'results_step7_positions.csv')
    parser.add_argument('--diagnostics', type=Path, default=HERE / 'results_step7_diagnostics.csv')
    parser.add_argument('--manifest', type=Path, default=HERE / 'step7_manifest.json')
    parser.add_argument('--batch-size', type=int, default=256)
    return parser.parse_args()


def main():
    args = parse_args()
    outputs = (args.results, args.positions, args.diagnostics, args.manifest, args.out_dir)
    if any(path.exists() for path in outputs):
        raise FileExistsError('Refusing to overwrite an existing Step 7 output')
    meta = load_meta(str(args.data_dir))
    expected_meta = dict(length=40, seen_end=10, data_seed=1337, counts=[0, 1, 2, 3], vocab_size=43)
    if any(meta[key] != value for key, value in expected_meta.items()):
        raise ValueError('This analysis requires the audited Step 5 dataset')
    expected_model = dict(vocab_size=43, block_size=96, n_layer=4, n_head=4, n_embd=128,
                          dropout=0.0, bias=True, pos_type='none', causal=True, t5_bias_mode='auto')
    checkpoints = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            path = Path(f'{args.checkpoint_prefix}_{condition}_{seed}') / 'ckpt_iter2000.pt'
            if not path.exists():
                raise FileNotFoundError(path)
            checkpoints[(condition, seed)] = path
    pools = build_pools(meta, args.data_dir)
    manifest = {
        'pool_seed': POOL_SEED, 'model_seeds': SEEDS, 'conditions': CONDITIONS,
        'probe': 'four-class ridge; fit-only standardization; unnormalized SSE + L2(1.0); free intercept',
        'primary_layer': 4, 'query': 'non-X character, followed by # when present',
        'torch_version': torch.__version__, 'numpy_version': np.__version__,
        'data_sha256': {name: checksum(args.data_dir / name)
                        for name in ('train.bin', 'val.bin', 'test.bin', 'meta.pkl')},
        'checkpoint_sha256': {str(path.relative_to(HERE)): checksum(path)
                              for path in checkpoints.values()},
        'pools': {},
    }
    for name, (examples, queries) in pools.items():
        payload = '\n'.join(f'{p}:{body}:{label}' for (body, label), p in zip(examples, queries))
        manifest['pools'][name] = dict(n=len(examples), positions=sorted(set(queries.tolist())),
                                     labels=dict(Counter(label for _, label in examples)),
                                     sha256=hashlib.sha256(payload.encode()).hexdigest())
    args.out_dir.mkdir(parents=True)
    result_rows, position_rows, diagnostic_rows = [], [], []
    for condition in CONDITIONS:
        tensors = {name: tensorize_examples(examples, condition, meta, name).inputs
                   for name, (examples, _) in pools.items()}
        for seed in SEEDS:
            checkpoint = torch.load(checkpoints[(condition, seed)], map_location='cpu', weights_only=False)
            if (checkpoint['model_args'] != expected_model
                    or any(checkpoint['data_meta'][key] != value for key, value in expected_meta.items())
                    or (checkpoint['condition'], checkpoint['seed'], checkpoint['iteration']) != (condition, seed, 2000)):
                raise ValueError(f'Checkpoint mismatch: {condition}, {seed}')
            for model_kind in ('trained', 'random'):
                torch.manual_seed(seed)
                model = MicroTransformer(MicroTransformerConfig(**expected_model)).eval()
                if model_kind == 'trained':
                    model.load_state_dict(checkpoint['model'])
                model.requires_grad_(False)
                frozen = {key: value.detach().clone() for key, value in model.state_dict().items()}
                extracted = {}
                for name, (examples, queries) in pools.items():
                    features, answer_hits, state_hits, total = feature_extract(
                        model, tensors[name], examples, queries, condition, meta, args.batch_size)
                    extracted[name] = features
                    diagnostic_rows.append(dict(condition=condition, seed=seed, model_kind=model_kind,
                        pool=name, n=total, answer_acc=answer_hits / total,
                        query_hash_prediction_acc=state_hits / total if condition == 'dummy' else ''))
                labels = {name: np.asarray([int(label) for _, label in examples])
                          for name, (examples, _) in pools.items()}
                shuffled = np.random.default_rng(POOL_SEED + seed).permutation(labels['fit'])
                saved = {}
                for site, layer in extracted['fit']:
                    for label_kind, fit_labels in (('true', labels['fit']), ('shuffled', shuffled)):
                        params = fit_ridge(extracted['fit'][(site, layer)], fit_labels)
                        stem = f'{site}_layer{layer}_{label_kind}'
                        for part, value in zip(('center', 'scale', 'weights', 'intercept'), params):
                            saved[f'{stem}_{part}'] = value
                        for name in ('seen', 'unseen'):
                            pred = ridge_predict(extracted[name][(site, layer)], params)
                            gold = labels[name]
                            recall = [float((pred[gold == k] == k).mean()) for k in range(4)]
                            row = dict(condition=condition, seed=seed, model_kind=model_kind, site=site,
                                layer=layer, labels=label_kind, pool=name, n=len(gold),
                                accuracy=float((pred == gold).mean()), macro_recall=float(np.mean(recall)))
                            row.update({f'recall_k{k}': value for k, value in enumerate(recall)})
                            result_rows.append(row)
                            if layer == 4:
                                for p in sorted(set(pools[name][1])):
                                    mask = pools[name][1] == p
                                    position_rows.append(dict(condition=condition, seed=seed, model_kind=model_kind,
                                        site=site, layer=layer, labels=label_kind, pool=name, position=int(p),
                                        n=int(mask.sum()), accuracy=float((pred[mask] == gold[mask]).mean())))
                np.savez_compressed(args.out_dir / f'{condition}_{seed}_{model_kind}.npz', **saved)
                if any(not torch.equal(value, frozen[key]) for key, value in model.state_dict().items()):
                    raise AssertionError('Frozen model changed')
                print(f'completed {condition} seed {seed} {model_kind}', flush=True)
                del extracted, model
    # Layer-0 control: count is balanced even conditional on query token identity.
    if any(abs(row['accuracy'] - 0.25) > 1e-10 for row in result_rows if row['layer'] == 0):
        raise AssertionError('Layer-0 control failed; check sampling or leakage')
    write_csv(args.results, result_rows)
    write_csv(args.positions, position_rows)
    write_csv(args.diagnostics, diagnostic_rows)
    args.manifest.write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'wrote {len(result_rows)} probe rows, {len(position_rows)} position rows', flush=True)


if __name__ == '__main__':
    main()
