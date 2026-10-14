"""Sequence construction and loss masks for the counting task.

The raw task is a fixed-length string of lowercase filler letters with k copies of
X; the answer is k.  This module is independent of PyTorch so data loading,
training, evaluation, and tests share one definition of each condition's format.

Formats (L = body length):

    no_scratchpad  mpXaX...:2
    dummy          #m#p#X#a#X#...#:2      '#' after every body token
    meaningful     0m0p0X1a1X2...2:2      running count of X after every body token

The leading '0' (or '#') is always the same, so it is given to the model rather than
predicted.  State targets are the L tokens written after the body tokens; the answer
target is the token after ':'.
"""
from dataclasses import dataclass
from typing import Dict, List


CONDITIONS = ('no_scratchpad', 'dummy', 'meaningful')
DUMMY_STATE = '#'
INITIAL_STATES = {'dummy': DUMMY_STATE, 'meaningful': '0'}
IGNORE_INDEX = -1


@dataclass(frozen=True)
class SupervisedExample:
    """One teacher-forced example with separate state and answer targets."""

    input_ids: List[int]
    state_targets: List[int]
    answer_targets: List[int]
    record: str


def validate_body(body: str) -> None:
    invalid = sorted(set(body) - set('abcdefghijklmnopqrstuvwyzX'))
    if invalid:
        raise ValueError(f"body contains invalid characters {invalid}: {body!r}")


def count_label(body: str) -> str:
    """The answer for a raw body: the number of X symbols, as one digit."""
    validate_body(body)
    return str(body.count('X'))


def check_condition(condition: str) -> None:
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")


def running_counts(body: str) -> List[str]:
    """Number of X symbols seen so far, after each body token."""
    validate_body(body)
    counts = []
    count = 0
    for token in body:
        count += token == 'X'
        counts.append(str(count))
    return counts


def initial_state_token(condition: str) -> str:
    """The given (not predicted) token before the first body token."""
    check_condition(condition)
    if condition == 'no_scratchpad':
        raise ValueError("the no-scratchpad condition has no scratchpad states")
    return INITIAL_STATES[condition]


def gold_state_tokens(body: str, condition: str) -> List[str]:
    """Gold state after every body token (empty for the no-scratchpad condition)."""
    check_condition(condition)
    if condition == 'no_scratchpad':
        validate_body(body)
        return []
    if condition == 'dummy':
        validate_body(body)
        return [DUMMY_STATE] * len(body)
    return running_counts(body)


def allowed_state_tokens(condition: str, count_tokens: List[str]) -> List[str]:
    check_condition(condition)
    if condition == 'meaningful':
        return list(count_tokens)
    if condition == 'dummy':
        return [DUMMY_STATE]
    return []


def build_supervised_example(
    body: str,
    label: str,
    condition: str,
    stoi: Dict[str, int],
) -> SupervisedExample:
    """Build one next-token input with disjoint state and answer loss masks.

    For a target at full record index j, the corresponding model logit is at input
    index j-1.  The no-scratchpad condition has an answer target only.
    """
    expected_label = count_label(body)
    if label != expected_label:
        raise ValueError(f"wrong label {label!r} for {body!r}; expected {expected_label!r}")
    check_condition(condition)

    tokens: List[str] = []
    roles: List[str] = []
    if condition == 'no_scratchpad':
        tokens.extend(body)
        roles.extend(['raw'] * len(body))
    else:
        tokens.append(initial_state_token(condition))
        roles.append('raw')
        for raw_token, state_token in zip(body, gold_state_tokens(body, condition)):
            tokens.extend((raw_token, state_token))
            roles.extend(('raw', 'state'))
    tokens.extend((':', label))
    roles.extend(('raw', 'answer'))

    missing = [token for token in tokens if token not in stoi]
    if missing:
        raise KeyError(f"tokens missing from vocabulary: {sorted(set(missing))}")

    encoded = [stoi[token] for token in tokens]
    input_ids = encoded[:-1]
    state_targets = [IGNORE_INDEX] * len(input_ids)
    answer_targets = [IGNORE_INDEX] * len(input_ids)

    for full_index in range(1, len(tokens)):
        if roles[full_index] == 'state':
            state_targets[full_index - 1] = encoded[full_index]
        elif roles[full_index] == 'answer':
            answer_targets[full_index - 1] = encoded[full_index]

    if sum(target != IGNORE_INDEX for target in answer_targets) != 1:
        raise AssertionError("every example must have exactly one answer target")
    expected_state_count = 0 if condition == 'no_scratchpad' else len(body)
    if sum(target != IGNORE_INDEX for target in state_targets) != expected_state_count:
        raise AssertionError("unexpected number of scratchpad targets")

    return SupervisedExample(
        input_ids=input_ids,
        state_targets=state_targets,
        answer_targets=answer_targets,
        record=''.join(tokens),
    )
