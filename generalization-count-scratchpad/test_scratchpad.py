"""Unit tests for count labels and no-scratchpad loss masks."""
import unittest

from scratchpad import (
    IGNORE_INDEX,
    build_supervised_example,
    check_condition,
    count_label,
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

    def test_scratchpad_conditions_not_built_yet(self):
        for condition in ('dummy', 'meaningful'):
            with self.assertRaises(NotImplementedError):
                check_condition(condition)


if __name__ == '__main__':
    unittest.main()
