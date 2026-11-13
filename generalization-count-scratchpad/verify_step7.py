"""Re-evaluate saved Step 7 readouts without fitting or changing any model.

Rebuild the recorded pools, manually encode the two input formats, observe normal
model forwards with independent hooks, and compare every stored probe, position,
and supplied-format diagnostic score. Existing result files are read-only.
"""
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from dataset import load_meta
from engine import MicroTransformer, MicroTransformerConfig
from probe_step7 import build_pools


HERE = Path(__file__).resolve().parent


def rows_by(path, fields):
    with path.open(newline='') as file:
        rows = list(csv.DictReader(file))
    indexed = {tuple(row[field] for field in fields): row for row in rows}
    assert len(indexed) == len(rows), f'Duplicate rows: {path}'
    return indexed


def equal_score(actual, expected, context):
    if abs(actual - float(expected)) > 1e-12:
        raise AssertionError(f'{context}: recomputed {actual}, stored {expected}')


def main():
    torch.set_num_threads(1)
    started = time.monotonic()
    manifest = json.loads((HERE / 'step7_manifest.json').read_text())
    data_dir = HERE / 'data/count_scratchpad_L40_seen10_step5'
    meta = load_meta(str(data_dir))
    pools = build_pools(meta, data_dir)
    probe_fields = ('condition', 'seed', 'model_kind', 'site', 'layer', 'labels', 'pool')
    probes = rows_by(HERE / 'results_step7_probes.csv', probe_fields)
    positions = rows_by(HERE / 'results_step7_positions.csv', probe_fields + ('position',))
    diagnostics = rows_by(HERE / 'results_step7_diagnostics.csv',
                          ('condition', 'seed', 'model_kind', 'pool'))
    checked_probes, checked_positions, checked_diagnostics = set(), set(), set()
    for pool, (examples, queries) in pools.items():
        payload = '\n'.join(f'{p}:{body}:{label}' for (body, label), p in zip(examples, queries))
        assert hashlib.sha256(payload.encode()).hexdigest() == manifest['pools'][pool]['sha256']

    for condition in manifest['conditions']:
        tensors = {}
        for pool, (examples, _) in pools.items():
            records = [body + ':' if condition == 'no_scratchpad'
                       else '#' + ''.join(char + '#' for char in body) + ':'
                       for body, _ in examples]
            tensors[pool] = torch.tensor([[meta['stoi'][char] for char in record]
                                          for record in records], dtype=torch.long)
            assert all(record.endswith(':') for record in records)
        for seed in manifest['model_seeds']:
            checkpoint_path = HERE / f'out_step5_nope_{condition}_{seed}/ckpt_iter2000.pt'
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            for model_kind in ('trained', 'random'):
                torch.manual_seed(seed)
                model = MicroTransformer(MicroTransformerConfig(**checkpoint['model_args'])).eval()
                if model_kind == 'trained':
                    model.load_state_dict(checkpoint['model'])
                model.requires_grad_(False)
                frozen = {key: value.clone() for key, value in model.state_dict().items()}
                with np.load(HERE / 'out_step7_probes_verified' /
                             f'{condition}_{seed}_{model_kind}.npz') as saved:
                    for pool, (examples, queries) in pools.items():
                        gold = np.asarray([int(label) for _, label in examples])
                        answer_ids = torch.tensor([meta['stoi'][label] for _, label in examples])
                        sites = {'raw': queries} if condition == 'no_scratchpad' else {
                            'raw': 2 * queries + 1, 'hash': 2 * queries + 2}
                        parts = {(site, layer): [] for site in sites for layer in range(5)}
                        active = {}

                        def make_hook(layer):
                            def hook(_module, _args, output):
                                start, end = active['bounds']
                                batch_rows = torch.arange(end - start)
                                for site, indices in sites.items():
                                    selected = output[batch_rows, torch.from_numpy(indices[start:end])]
                                    parts[(site, layer)].append(selected.detach().clone())
                            return hook

                        hooks = [model.drop.register_forward_hook(make_hook(0))]
                        hooks += [block.register_forward_hook(make_hook(layer))
                                  for layer, block in enumerate(model.blocks, 1)]
                        answer_hits = hash_hits = 0
                        try:
                            with torch.inference_mode():
                                for start in range(0, len(examples), 256):
                                    end = min(start + 256, len(examples))
                                    active['bounds'] = (start, end)
                                    logits = model(tensors[pool][start:end])
                                    answer_hits += int((logits[:, -1].argmax(-1) == answer_ids[start:end]).sum())
                                    if condition == 'dummy':
                                        raw_sites = torch.from_numpy(sites['raw'][start:end])
                                        predicted = logits[torch.arange(end - start), raw_sites].argmax(-1)
                                        hash_hits += int((predicted == meta['stoi']['#']).sum())
                        finally:
                            for handle in hooks:
                                handle.remove()
                        diagnostic_key = (condition, str(seed), model_kind, pool)
                        row = diagnostics[diagnostic_key]
                        equal_score(answer_hits / len(examples), row['answer_acc'], diagnostic_key)
                        if condition == 'dummy':
                            equal_score(hash_hits / len(examples), row['query_hash_prediction_acc'], diagnostic_key)
                        checked_diagnostics.add(diagnostic_key)
                        if pool == 'fit':
                            # Extraction for fit is checked through the saved feature
                            # standardization; no readout is fitted or selected here.
                            fit_features = {key: torch.cat(chunks).numpy().astype(np.float64)
                                            for key, chunks in parts.items()}
                            for (site, layer), features in fit_features.items():
                                expected_mean = features.mean(axis=0)
                                expected_scale = features.std(axis=0)
                                expected_scale[expected_scale < 1e-8] = 1.0
                                for labels in ('true', 'shuffled'):
                                    stem = f'{site}_layer{layer}_{labels}'
                                    np.testing.assert_allclose(expected_mean, saved[f'{stem}_center'], atol=1e-9, rtol=1e-7)
                                    np.testing.assert_allclose(expected_scale, saved[f'{stem}_scale'], atol=1e-9, rtol=1e-7)
                            continue
                        for (site, layer), chunks in parts.items():
                            features = torch.cat(chunks).to(torch.float64)
                            for labels in ('true', 'shuffled'):
                                stem = f'{site}_layer{layer}_{labels}'
                                parameters = [torch.from_numpy(saved[f'{stem}_{part}'])
                                              for part in ('center', 'scale', 'weights', 'intercept')]
                                center, scale, weights, intercept = parameters
                                scores = ((features - center) / scale) @ weights + intercept
                                assert torch.isfinite(scores).all()
                                pred = scores.argmax(-1).numpy()
                                key = (condition, str(seed), model_kind, site, str(layer), labels, pool)
                                row = probes[key]
                                recall = [float((pred[gold == count] == count).mean()) for count in range(4)]
                                equal_score(float((pred == gold).mean()), row['accuracy'], key)
                                equal_score(sum(recall) / 4, row['macro_recall'], key)
                                for count, score in enumerate(recall):
                                    equal_score(score, row[f'recall_k{count}'], key + (str(count),))
                                checked_probes.add(key)
                                if layer == 4:
                                    for query in sorted(set(queries)):
                                        mask = queries == query
                                        position_key = key + (str(query),)
                                        equal_score(float((pred[mask] == gold[mask]).mean()),
                                                    positions[position_key]['accuracy'], position_key)
                                        checked_positions.add(position_key)
                assert all(torch.equal(value, frozen[key]) for key, value in model.state_dict().items())
                print(f'verified {condition} seed {seed} {model_kind}; '
                      f'elapsed {time.monotonic() - started:.1f}s', flush=True)
    assert checked_probes == set(probes)
    assert checked_positions == set(positions)
    assert checked_diagnostics == set(diagnostics)
    print(f'PASS: independently reproduced all {len(probes)} probe rows, '
          f'{len(positions)} position rows and {len(diagnostics)} diagnostic rows '
          'from saved checkpoints and readouts, without refitting or changing weights.')


if __name__ == '__main__':
    main()
