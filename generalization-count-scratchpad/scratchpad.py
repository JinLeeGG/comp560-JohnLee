"""Sequence construction and loss masks for the counting task.

The raw task is a fixed-length string of lowercase filler letters with k copies of
X; the answer is k.  This module is independent of PyTorch so data loading,
training, evaluation, and tests share one definition of each condition's format.

Step 1 (baseline gate) implements only the no-scratchpad condition.  The dummy and
meaningful formats from the plan are added in Step 3, after the gate.
"""
from dataclasses import dataclass
from typing import Dict, List


CONDITIONS = ('no_scratchpad', 'dummy', 'meaningful')
IMPLEMENTED_CONDITIONS = ('no_scratchpad',)
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
    if condition not in IMPLEMENTED_CONDITIONS:
        raise NotImplementedError(
            f"condition {condition!r} is planned for Step 3 and not implemented yet"
        )


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

    tokens = list(body) + [':', label]
    roles = ['raw'] * (len(body) + 1) + ['answer']

    missing = [token for token in tokens if token not in stoi]
    if missing:
        raise KeyError(f"tokens missing from vocabulary: {sorted(set(missing))}")

    encoded = [stoi[token] for token in tokens]
    input_ids = encoded[:-1]
    state_targets = [IGNORE_INDEX] * len(input_ids)
    answer_targets = [IGNORE_INDEX] * len(input_ids)

    for full_index in range(1, len(tokens)):
        if roles[full_index] == 'answer':
            answer_targets[full_index - 1] = encoded[full_index]

    if sum(target != IGNORE_INDEX for target in answer_targets) != 1:
        raise AssertionError("every example must have exactly one answer target")

    return SupervisedExample(
        input_ids=input_ids,
        state_targets=state_targets,
        answer_targets=answer_targets,
        record=''.join(tokens),
    )
