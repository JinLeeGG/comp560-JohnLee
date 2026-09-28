"""Unit tests for state transitions, loss masks, and gold chained contexts."""
import unittest

from scratchpad import (
    IGNORE_INDEX,
    build_supervised_example,
    distance_and_label,
    gold_state_tokens,
    meaningful_states,
)


VOCAB = sorted(set('0123456789XTF:\niabcdef tz'.replace(' ', '')))
STOI = {token: index for index, token in enumerate(VOCAB)}
ITOS = {index: token for token, index in STOI.items()}


class StateMachineTests(unittest.TestCase):
    def test_far_boundary_distance_five(self):
        body = '2X1234X7'
        self.assertEqual(distance_and_label(body), (5, 'T'))
        self.assertEqual(''.join(meaningful_states(body)), 'iabcdett')
        self.assertEqual(''.join(gold_state_tokens(body, 'meaningful')), 'iabcdettt')

    def test_near_boundary_distance_four(self):
        body = '2X123X7'
        self.assertEqual(distance_and_label(body), (4, 'F'))
        self.assertEqual(''.join(meaningful_states(body)), 'iabcdff')

    def test_saturated_far_state(self):
        body = 'X12345678X'
        states = meaningful_states(body)
        self.assertEqual(states, list('abcdeeeeet'))
        self.assertEqual(distance_and_label(body), (9, 'T'))

    def test_adjacent_x_is_near(self):
        body = '12XX34'
        self.assertEqual(distance_and_label(body), (1, 'F'))
        self.assertEqual(meaningful_states(body), list('iiafff'))


class SupervisionTests(unittest.TestCase):
    def decoded_target(self, target):
        return None if target == IGNORE_INDEX else ITOS[target]

    def assert_targets_match_record(self, example):
        for index, target in enumerate(example.state_targets):
            if target != IGNORE_INDEX:
                self.assertEqual(self.decoded_target(target), example.record[index + 1])
        for index, target in enumerate(example.answer_targets):
            if target != IGNORE_INDEX:
                self.assertEqual(self.decoded_target(target), example.record[index + 1])

    def test_meaningful_record_and_masks(self):
        example = build_supervised_example('2X1234X7', 'T', 'meaningful', STOI)
        self.assertEqual(example.record, '2iXa1b2c3d4eXt7t:tT')
        self.assertEqual(len(example.input_ids), 18)
        self.assertEqual(
            sum(target != IGNORE_INDEX for target in example.state_targets), 9
        )
        self.assertEqual(
            sum(target != IGNORE_INDEX for target in example.answer_targets), 1
        )
        self.assert_targets_match_record(example)

    def test_dummy_is_length_matched(self):
        meaningful = build_supervised_example('2X1234X7', 'T', 'meaningful', STOI)
        dummy = build_supervised_example('2X1234X7', 'T', 'dummy', STOI)
        self.assertEqual(dummy.record, '2zXz1z2z3z4zXz7z:zT')
        self.assertEqual(len(dummy.input_ids), len(meaningful.input_ids))
        self.assertEqual(
            sum(target != IGNORE_INDEX for target in dummy.state_targets), 9
        )
        self.assert_targets_match_record(dummy)

    def test_no_scratchpad_has_answer_only_loss(self):
        example = build_supervised_example('2X1234X7', 'T', 'no_scratchpad', STOI)
        self.assertEqual(example.record, '2X1234X7:T')
        self.assertEqual(len(example.input_ids), 9)
        self.assertTrue(all(target == IGNORE_INDEX for target in example.state_targets))
        self.assertEqual(
            sum(target != IGNORE_INDEX for target in example.answer_targets), 1
        )
        self.assert_targets_match_record(example)

    def test_gold_chain_context_is_teacher_forced_prompt(self):
        for condition in ('dummy', 'meaningful'):
            example = build_supervised_example('2X1234X7', 'T', condition, STOI)
            decoded_input = ''.join(ITOS[token_id] for token_id in example.input_ids)
            self.assertEqual(decoded_input, example.record[:-1])

    def test_wrong_label_is_rejected(self):
        with self.assertRaises(ValueError):
            build_supervised_example('2X1234X7', 'F', 'meaningful', STOI)


if __name__ == '__main__':
    unittest.main()
