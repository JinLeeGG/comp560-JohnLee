"""Unit tests for count labels, scratchpad formats, and loss masks."""
import random
import unittest

from dataset import example_fillers
from scratchpad import (
    GIVEN_FILLER_CONDITIONS,
    IGNORE_INDEX,
    build_supervised_example,
    check_condition,
    count_label,
    draw_fillers,
    gold_state_tokens,
    running_counts,
)


FILLER = 'abcdefghijklmnopqrstuvwyz'
VOCAB = sorted(set(FILLER + 'X0123#:\n'))
STOI = {token: index for index, token in enumerate(VOCAB)}
ITOS = {index: token for token, index in STOI.items()}

RANDOM_SYMBOLS = '!$%&*+<=>?'
STEP5_VOCAB = sorted(set(VOCAB) | set(RANDOM_SYMBOLS))
STEP5_STOI = {token: index for index, token in enumerate(STEP5_VOCAB)}
STEP5_ITOS = {index: token for token, index in STEP5_STOI.items()}


def step5_example(body, label, condition, rng=None):
    fillers = draw_fillers(condition, len(body), rng or random.Random(0), RANDOM_SYMBOLS)
    return build_supervised_example(body, label, condition, STEP5_STOI, fillers)


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


class Step5FormatTests(unittest.TestCase):
    BODY = 'mpXaXdqXrbkzuwhnfeoc'

    def test_records_match_plan_formats(self):
        expected = {
            'pad_end': self.BODY + '#' * 21 + ':3',
            'pad_start': '#' * 21 + self.BODY + ':3',
            'every2': '#mp#Xa#Xd#qX#rb#kz#uw#hn#fe#oc#:3',
            'every4': '#mpXa#XdqX#rbkz#uwhn#feoc#:3',
        }
        for condition, record in expected.items():
            self.assertEqual(step5_example(self.BODY, '3', condition).record, record)
        self.assertEqual(
            step5_example(self.BODY, '3', 'pad_end_given').record, expected['pad_end']
        )

    def test_written_versions_write_every_hash_but_the_first_token(self):
        # 20-letter body: pad_end writes all 21 '#'; the others are given their first.
        expected_states = {'pad_end': 21, 'pad_start': 20, 'every2': 10, 'every4': 5}
        for condition, count in expected_states.items():
            example = step5_example(self.BODY, '3', condition)
            state_positions = [
                index for index, target in enumerate(example.state_targets)
                if target != IGNORE_INDEX
            ]
            self.assertEqual(len(state_positions), count)
            # The state at record index j is predicted from input index j-1.
            self.assertTrue(all(example.record[i + 1] == '#' for i in state_positions))
            self.assertEqual(gold_state_tokens(self.BODY, condition), ['#'] * count)
            self.assertEqual(STEP5_ITOS[example.answer_targets[-1]], '3')

    def test_written_dummy_layouts_match_the_dummy_masks(self):
        # Every '#' after a body token is a target, as in the Step 3 dummy.
        dummy = build_supervised_example(self.BODY, '3', 'dummy', STEP5_STOI)
        every2 = step5_example(self.BODY, '3', 'every2')
        pad_start = step5_example(self.BODY, '3', 'pad_start')
        self.assertEqual(dummy.state_targets[0], IGNORE_INDEX)
        self.assertEqual(every2.state_targets[0], IGNORE_INDEX)
        # pad_start writes '#' 2..21 from the '#' before each, then reads the body.
        self.assertEqual(
            [i for i, t in enumerate(pad_start.state_targets) if t != IGNORE_INDEX],
            list(range(20)),
        )

    def test_dummy_given_has_the_dummy_record(self):
        dummy = build_supervised_example(self.BODY, '3', 'dummy', STEP5_STOI)
        given = step5_example(self.BODY, '3', 'dummy_given')
        self.assertEqual(given.record, dummy.record)
        self.assertEqual(given.input_ids, dummy.input_ids)

    def test_random_filler_keeps_the_dummy_layout(self):
        record = step5_example(self.BODY, '3', 'random_filler').record
        self.assertEqual(record[1:-2:2], self.BODY)
        slots = record[0:-2:2]
        self.assertEqual(len(slots), 21)
        self.assertTrue(set(slots) <= set(RANDOM_SYMBOLS))
        self.assertTrue(record.endswith(':3'))

    def test_given_versions_have_only_the_answer_target(self):
        for condition in GIVEN_FILLER_CONDITIONS:
            example = step5_example(self.BODY, '3', condition)
            self.assertTrue(all(t == IGNORE_INDEX for t in example.state_targets))
            answer_positions = [
                index for index, target in enumerate(example.answer_targets)
                if target != IGNORE_INDEX
            ]
            self.assertEqual(answer_positions, [len(example.input_ids) - 1])
            self.assertEqual(STEP5_ITOS[example.input_ids[-1]], ':')
            self.assertEqual(STEP5_ITOS[example.answer_targets[-1]], '3')
            self.assertEqual(gold_state_tokens(self.BODY, condition), [])

    def test_length_40_sequence_lengths(self):
        body = 'a' * 15 + 'X' + 'b' * 24
        expected = {
            'dummy_given': 82, 'pad_end': 82, 'pad_start': 82, 'pad_end_given': 82,
            'every2': 62, 'every4': 52, 'random_filler': 82,
        }
        for condition, length in expected.items():
            self.assertEqual(len(step5_example(body, '1', condition).input_ids), length)

    def test_random_fillers_repeat_with_the_seed(self):
        first = draw_fillers('random_filler', 40, random.Random(7), RANDOM_SYMBOLS)
        second = draw_fillers('random_filler', 40, random.Random(7), RANDOM_SYMBOLS)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 41)
        self.assertGreater(len(set(first)), 1)

    def test_example_fillers_are_fixed_per_split(self):
        meta = {'data_seed': 1337, 'random_symbols': RANDOM_SYMBOLS}
        examples = [(self.BODY, '3'), ('abcdefghijklmnopqrst', '0')]
        train = example_fillers(examples, 'random_filler', meta, 'train')
        self.assertEqual(train, example_fillers(examples, 'random_filler', meta, 'train'))
        self.assertNotEqual(train, example_fillers(examples, 'random_filler', meta, 'test'))
        self.assertEqual(example_fillers(examples, 'dummy', meta, 'train'), [None, None])
        self.assertEqual(
            example_fillers(examples, 'pad_end', meta, 'train'), [['#'] * 21] * 2
        )

    def test_misuse_is_rejected(self):
        with self.assertRaises(ValueError):
            build_supervised_example(self.BODY, '3', 'pad_end', STEP5_STOI)
        with self.assertRaises(ValueError):
            build_supervised_example(self.BODY, '3', 'dummy', STEP5_STOI, ['#'] * 21)
        with self.assertRaises(ValueError):
            draw_fillers('every4', 42, random.Random(0), RANDOM_SYMBOLS)
        with self.assertRaises(ValueError):
            draw_fillers('random_filler', 40, random.Random(0), '')


if __name__ == '__main__':
    unittest.main()
