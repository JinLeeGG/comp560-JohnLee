"""Shared scratchpad state machine and supervised-sequence construction.

The raw task is fixed-length dist>=5 with exactly two X symbols.  This module is
deliberately independent of PyTorch so data preparation, training, evaluation, and
tests all use one definition of the scratchpad algorithm and loss-mask positions.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence


CONDITIONS = ('no_scratchpad', 'dummy', 'meaningful')
MEANINGFUL_STATES = ('i', 'a', 'b', 'c', 'd', 'e', 't', 'f')
DUMMY_STATE = 'z'
IGNORE_INDEX = -1


@dataclass(frozen=True)
class SupervisedExample:
    """One teacher-forced example with separate state and answer targets."""

    input_ids: List[int]
    state_targets: List[int]
    answer_targets: List[int]
    record: str


def validate_body(body: str) -> None:
    if body.count('X') != 2:
        raise ValueError(f"body must contain exactly two X symbols: {body!r}")
    invalid = set(body) - set('0123456789X')
    if invalid:
        raise ValueError(f"body contains invalid characters {sorted(invalid)}: {body!r}")


def distance_and_label(body: str, threshold: int = 5):
    """Return (distance, label) for a valid raw body."""
    validate_body(body)
    positions = [i for i, char in enumerate(body) if char == 'X']
    distance = positions[1] - positions[0]
    return distance, ('T' if distance >= threshold else 'F')


def meaningful_states(body: str, threshold: int = 5) -> List[str]:
    """State after each raw body token.

    For threshold 5, `a` means zero intervening tokens after the first X,
    `b`/`c`/`d` mean one/two/three, and `e` means four or more.  Encountering
    the second X from `e` yields `t`; encountering it from an earlier state
    yields `f`.  Terminal states persist.
    """
    if threshold != 5:
        raise ValueError("the current finite-state encoding is defined only for threshold=5")
    validate_body(body)

    state = 'i'
    states = []
    advance = {'a': 'b', 'b': 'c', 'c': 'd', 'd': 'e', 'e': 'e'}

    for token in body:
        if state == 'i':
            state = 'a' if token == 'X' else 'i'
        elif state in advance:
            if token == 'X':
                state = 't' if state == 'e' else 'f'
            else:
                state = advance[state]
        elif state in ('t', 'f'):
            # The result is final once the second X has been processed.
            state = state
        else:  # pragma: no cover - defensive guard for future edits
            raise AssertionError(f"unknown scratchpad state: {state!r}")
        states.append(state)

    _, label = distance_and_label(body, threshold)
    expected_terminal = label.lower()
    if states[-1] != expected_terminal:
        raise AssertionError(
            f"state machine ended at {states[-1]!r}, expected {expected_terminal!r}"
        )
    return states


def gold_state_tokens(body: str, condition: str, threshold: int = 5) -> List[str]:
    """Gold states after every body token plus one state after the ':' delimiter."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    if condition == 'no_scratchpad':
        return []
    if condition == 'dummy':
        return [DUMMY_STATE] * (len(body) + 1)
    body_states = meaningful_states(body, threshold)
    return body_states + [body_states[-1]]


def allowed_state_tokens(condition: str) -> Sequence[str]:
    if condition == 'meaningful':
        return MEANINGFUL_STATES
    if condition == 'dummy':
        return (DUMMY_STATE,)
    if condition == 'no_scratchpad':
        return ()
    raise ValueError(f"unknown condition: {condition!r}")


def build_supervised_example(
    body: str,
    label: str,
    condition: str,
    stoi: Dict[str, int],
    threshold: int = 5,
) -> SupervisedExample:
    """Build one next-token input with disjoint state and answer loss masks.

    Roles are attached to target tokens in the full record.  For a target at full
    record index j, the corresponding model logit is at input index j-1.
    """
    distance, expected_label = distance_and_label(body, threshold)
    if label != expected_label:
        raise ValueError(
            f"wrong label {label!r} for distance {distance}; expected {expected_label!r}"
        )
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")

    tokens: List[str] = []
    roles: List[str] = []

    if condition == 'no_scratchpad':
        tokens.extend(body)
        roles.extend(['raw'] * len(body))
        tokens.append(':')
        roles.append('raw')
    else:
        raw_prompt = body + ':'
        states = gold_state_tokens(body, condition, threshold)
        for raw_token, state_token in zip(raw_prompt, states):
            tokens.extend((raw_token, state_token))
            roles.extend(('raw', 'state'))

    tokens.append(label)
    roles.append('answer')

    missing = [token for token in tokens if token not in stoi]
    if missing:
        raise KeyError(f"tokens missing from vocabulary: {sorted(set(missing))}")

    encoded = [stoi[token] for token in tokens]
    input_ids = encoded[:-1]
    state_targets = [IGNORE_INDEX] * len(input_ids)
    answer_targets = [IGNORE_INDEX] * len(input_ids)

    for full_index in range(1, len(tokens)):
        target_index = full_index - 1
        if roles[full_index] == 'state':
            state_targets[target_index] = encoded[full_index]
        elif roles[full_index] == 'answer':
            answer_targets[target_index] = encoded[full_index]

    if sum(target != IGNORE_INDEX for target in answer_targets) != 1:
        raise AssertionError("every example must have exactly one answer target")
    expected_state_count = 0 if condition == 'no_scratchpad' else len(body) + 1
    if sum(target != IGNORE_INDEX for target in state_targets) != expected_state_count:
        raise AssertionError("unexpected number of scratchpad targets")

    return SupervisedExample(
        input_ids=input_ids,
        state_targets=state_targets,
        answer_targets=answer_targets,
        record=''.join(tokens),
    )
