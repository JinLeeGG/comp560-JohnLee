"""Sampling, readout and causal-leakage checks for the frozen-state diagnostic."""
import unittest

import numpy as np
import torch

from dataset import tensorize_examples
from engine import MicroTransformer, MicroTransformerConfig
from probe_step7 import feature_extract, fit_ridge, make_pool, ridge_predict, site_indices


TOKENS = sorted(set('abcdefghijklmnopqrstuvwyzX0123#:\n!$%&*+<=>?'))
META = dict(filler='abcdefghijklmnopqrstuvwyz', data_seed=1337,
            stoi={token: i for i, token in enumerate(TOKENS)}, random_symbols='!$%&*+<=>?')


class ProbeTests(unittest.TestCase):
    def test_balanced_pools_preserve_split_and_match_query_identity(self):
        forbidden = set()
        seen, seen_q = make_pool(META, 'fit', range(3, 10), 2, forbidden)
        unseen, unseen_q = make_pool(META, 'unseen', range(13, 40), 2, forbidden)
        self.assertEqual(len(forbidden), len(seen) + len(unseen))
        for examples, queries, low in ((seen, seen_q, 0), (unseen, unseen_q, 10)):
            for start in range(0, len(examples), 4):
                quartet = examples[start:start + 4]
                p = queries[start]
                self.assertEqual([label for _, label in quartet], ['0', '1', '2', '3'])
                self.assertEqual(len({body[p] for body, _ in quartet}), 1)
                for body, label in quartet:
                    self.assertEqual(len(body), 40)
                    self.assertNotEqual(body[p], 'X')
                    self.assertEqual(body[:p + 1].count('X'), int(label))
                    self.assertEqual(body.count('X'), int(label))
                    self.assertTrue(all(low <= i < p for i, char in enumerate(body) if char == 'X'))

    def test_impossible_balancing_positions_are_rejected(self):
        with self.assertRaises(ValueError):
            make_pool(META, 'unseen', [12], 1, set())

    def test_ridge_reads_linear_signal_and_handles_constant_features(self):
        labels = np.tile(np.arange(4), 8)
        x = np.column_stack((np.eye(4)[labels], np.ones(len(labels))))
        params = fit_ridge(x, labels)
        np.testing.assert_array_equal(ridge_predict(x, params), labels)
        constant = np.ones((32, 5))
        pred = ridge_predict(constant, fit_ridge(constant, labels))
        self.assertEqual(float((pred == labels).mean()), 0.25)

    def test_future_tokens_cannot_change_query_representation(self):
        torch.manual_seed(7)
        model = MicroTransformer(MicroTransformerConfig(
            vocab_size=len(TOKENS), block_size=96, n_layer=4, n_head=2,
            n_embd=16, pos_type='none', dropout=0.0, causal=True)).eval()
        body = 'X' + 'a' * 39
        modified = body[:30] + 'b' * 10
        examples = [(body, '1'), (modified, '1')]
        queries = np.array([5, 5])
        for condition in ('no_scratchpad', 'dummy', 'dummy_given'):
            split = tensorize_examples(examples, condition, META, 'seen')
            self.assertEqual(int(split.inputs[0, -1]), META['stoi'][':'])
            features, _, _, total = feature_extract(
                model, split.inputs, examples, queries, condition, META, 2)
            self.assertEqual(total, 2)
            for matrix in features.values():
                np.testing.assert_allclose(matrix[0], matrix[1], rtol=0, atol=1e-6)
            if condition != 'no_scratchpad':
                np.testing.assert_allclose(features[('hash', 0)][0],
                    model.tok_emb.weight[META['stoi']['#']].detach().numpy(), atol=1e-7)
            self.assertFalse(model.drop._forward_hooks)
            self.assertTrue(all(not block._forward_hooks for block in model.blocks))


if __name__ == '__main__':
    unittest.main()
