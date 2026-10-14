"""Unit tests for count labels, scratchpad formats, and loss masks."""
import unittest

from scratchpad import (
    IGNORE_INDEX,
    build_supervised_example,
    check_condition,
    count_label,
    gold_state_tokens,
    running_counts,
)


FILLER = 'abcdefghijklmnopqrstuvwyz'
VOCAB = sorted(set(FILLER + 'X0123#:\n'))
STOI = {token: index for index, token in enumerate(VOCAB)}
ITOS = {index: token for token, index in STOI.items()}


class CountLabelTests(unittest.TestCase):
    def test_counts_uppercase_x_only(self):
        self.assertEqual(count_label('mpXaXdqXrbkzuwhnfeoc'), '3')
        self.assertEqual(count_label('abcdefghij'), '0')

    def test_rejects_lowercase_x_and_digits(self):
        with self.assertRaises(ValueError):
            count_label('abxX')
        with self.assertRaises(ValueError):
            count_label('ab1X')


class SupervisionTests(unittest.TestCase):
    def test_no_scratchpad_record_and_mask(self):
        example = build_supervised_example('mpXaXdqXrbkzuwhnfeoc', '3', 'no_scratchpad', STOI)
        self.assertEqual(example.record, 'mpXaXdqXrbkzuwhnfeoc:3')
        self.assertEqual(len(example.input_ids), 21)
        self.assertTrue(all(target == IGNORE_INDEX for target in example.state_targets))
        answer_positions = [
            index for index, target in enumerate(example.answer_targets)
            if target != IGNORE_INDEX
        ]
        # The answer is predicted from the ':' position, the last input token.
        self.assertEqual(answer_positions, [20])
        self.assertEqual(ITOS[example.answer_targets[20]], '3')
        self.assertEqual(ITOS[example.input_ids[20]], ':')

    def test_wrong_label_is_rejected(self):
        with self.assertRaises(ValueError):
            build_supervised_example('mpXaXdqXrbkzuwhnfeoc', '2', 'no_scratchpad', STOI)

    def test_unknown_condition_is_rejected(self):
        with self.assertRaises(ValueError):
            check_condition('scratch')


class ScratchpadFormatTests(unittest.TestCase):
    BODY = 'mpXaXdqXrbkzuwhnfeoc'

    def test_running_counts_increment_at_each_x(self):
        self.assertEqual(
            ''.join(running_counts(self.BODY)), '00112223333333333333'
        )
        self.assertEqual(running_counts('abc'), ['0', '0', '0'])

    def test_meaningful_record_matches_plan_format(self):
        example = build_supervised_example(self.BODY, '3', 'meaningful', STOI)
        self.assertTrue(example.record.startswith('0m0p0X1a1X2d2q2X3r3b3k3'))
        self.assertTrue(example.record.endswith('c3:3'))
        # Given '0', then (body token, state) pairs, then ':' and the answer.
        self.assertEqual(len(example.record), 2 * len(self.BODY) + 3)

    def test_dummy_record_matches_plan_format(self):
        example = build_supervised_example(self.BODY, '3', 'dummy', STOI)
        self.assertTrue(example.record.startswith('#m#p#X#a#X#'))
        self.assertTrue(example.record.endswith('c#:3'))
        self.assertEqual(len(example.record), 2 * len(self.BODY) + 3)

    def test_state_and_answer_masks(self):
        for condition in ('dummy', 'meaningful'):
            example = build_supervised_example(self.BODY, '3', condition, STOI)
            state_positions = [
                index for index, target in enumerate(example.state_targets)
                if target != IGNORE_INDEX
            ]
            # Each state is predicted from the body token just before it.
            self.assertEqual(state_positions, list(range(1, 2 * len(self.BODY), 2)))
            self.assertTrue(all(
                ITOS[example.input_ids[index]] in FILLER + 'X' for index in state_positions
            ))
            predicted_states = [ITOS[example.state_targets[i]] for i in state_positions]
            self.assertEqual(predicted_states, gold_state_tokens(self.BODY, condition))
            answer_positions = [
                index for index, target in enumerate(example.answer_targets)
                if target != IGNORE_INDEX
            ]
            self.assertEqual(answer_positions, [len(example.input_ids) - 1])
            self.assertEqual(ITOS[example.input_ids[-1]], ':')
            self.assertEqual(ITOS[example.answer_targets[-1]], '3')
            # The given leading token is never a target.
            self.assertEqual(example.state_targets[0], IGNORE_INDEX)

    def test_masks_are_disjoint(self):
        example = build_supervised_example(self.BODY, '3', 'meaningful', STOI)
        for state, answer in zip(example.state_targets, example.answer_targets):
            self.assertFalse(state != IGNORE_INDEX and answer != IGNORE_INDEX)

    def test_length_40_sequence_fits_block_96(self):
        body = 'a' * 15 + 'X' + 'b' * 24
        for condition in ('dummy', 'meaningful'):
            example = build_supervised_example(body, '1', condition, STOI)
            self.assertEqual(len(example.input_ids), 82)


if __name__ == '__main__':
    unittest.main()
