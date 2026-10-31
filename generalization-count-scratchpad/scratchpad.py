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

Step 5 moves or thins out the dummy's '#' slots.  As with the dummy, the model writes
every '#' itself (state targets, free-running at test), except the first token of a
sequence, which nothing comes before and so is given:

    pad_end        mpXaX...###...###:2    L+1 '#' after the body, all written
    pad_start      ###...###mpXaX...:2    L+1 '#' before the body, first one given
    every2         #mp#Xa#X...#:2         '#' after every 2nd body token
    every4         #mpXa#X...#:2          '#' after every 4th body token

Random symbols cannot be written by the model, so random_filler gives them and has no
loss on them; dummy_given is the dummy record under the same rule, its reference.
pad_end_given is pad_end under that rule (a stopped first batch):

    dummy_given    #m#p#X#a#X#...#:2      same record as dummy
    random_filler  !m%p+X&a?X...<:2       dummy layout, a random symbol in every slot
    pad_end_given  mpXaX...###...###:2    same record as pad_end
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple


WRITTEN_FILLER_CONDITIONS = ('pad_end', 'pad_start', 'every2', 'every4')
GIVEN_FILLER_CONDITIONS = ('dummy_given', 'random_filler', 'pad_end_given')
FILLER_CONDITIONS = WRITTEN_FILLER_CONDITIONS + GIVEN_FILLER_CONDITIONS
CONDITIONS = ('no_scratchpad', 'dummy', 'meaningful') + FILLER_CONDITIONS
DUMMY_STATE = '#'
INITIAL_STATES = {'dummy': DUMMY_STATE, 'meaningful': '0'}
# Where a Step 5 version puts its filler slots: an interleave stride, or one block.
FILLER_LAYOUTS = {
    'pad_end': 'end', 'pad_start': 'start', 'every2': 2, 'every4': 4,
    'dummy_given': 1, 'random_filler': 1, 'pad_end_given': 'end',
}
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
    if condition not in INITIAL_STATES:
        raise ValueError(f"the {condition} condition has no scratchpad states")
    return INITIAL_STATES[condition]


def gold_state_tokens(body: str, condition: str) -> List[str]:
    """Gold states the model writes, in order (empty when it writes none)."""
    check_condition(condition)
    validate_body(body)
    if condition in WRITTEN_FILLER_CONDITIONS:
        _, roles = filler_layout(body, condition, ['#'] * filler_count(condition, len(body)))
        return [DUMMY_STATE] * roles.count('state')
    if condition not in INITIAL_STATES:
        return []
    if condition == 'dummy':
        return [DUMMY_STATE] * len(body)
    return running_counts(body)


def allowed_state_tokens(condition: str, count_tokens: List[str]) -> List[str]:
    check_condition(condition)
    if condition == 'meaningful':
        return list(count_tokens)
    if condition == 'dummy' or condition in WRITTEN_FILLER_CONDITIONS:
        return [DUMMY_STATE]
    return []


def filler_count(condition: str, length: int) -> int:
    """Number of filler tokens a Step 5 version adds to a body of this length."""
    if condition not in FILLER_CONDITIONS:
        raise ValueError(f"the {condition} condition has no filler tokens")
    layout = FILLER_LAYOUTS[condition]
    if layout in ('end', 'start'):
        # As many as the dummy has, so only their place changes.
        return length + 1
    if length % layout:
        raise ValueError(f"{condition} needs a length divisible by {layout}, got {length}")
    return length // layout + 1


def draw_fillers(condition: str, length: int, rng, random_symbols: str) -> List[str]:
    """The filler tokens for one example: '#', or random symbols for random_filler."""
    count = filler_count(condition, length)
    if condition != 'random_filler':
        return [DUMMY_STATE] * count
    if not random_symbols:
        raise ValueError("random_filler needs data built with --random_symbols")
    return [rng.choice(random_symbols) for _ in range(count)]


def filler_layout(
    body: str, condition: str, fillers: Sequence[str]
) -> Tuple[List[str], List[str]]:
    """Body and filler tokens in reading order before ':', with their roles.

    A filler the model writes is a 'state'; the first token of the sequence and every
    filler of a given version are 'raw' (supplied, no loss).
    """
    validate_body(body)
    if len(fillers) != filler_count(condition, len(body)):
        raise ValueError(f"{condition} needs {filler_count(condition, len(body))} fillers")
    layout = FILLER_LAYOUTS[condition]
    if layout == 'end':
        tokens = list(body) + list(fillers)
        is_filler = [False] * len(body) + [True] * len(fillers)
    elif layout == 'start':
        tokens = list(fillers) + list(body)
        is_filler = [True] * len(fillers) + [False] * len(body)
    else:
        tokens, is_filler = [fillers[0]], [True]
        for group, start in enumerate(range(0, len(body), layout), start=1):
            chunk = body[start:start + layout]
            tokens.extend(chunk)
            tokens.append(fillers[group])
            is_filler.extend([False] * len(chunk) + [True])
    written = condition in WRITTEN_FILLER_CONDITIONS
    roles = [
        'state' if written and filler and index > 0 else 'raw'
        for index, filler in enumerate(is_filler)
    ]
    return tokens, roles


def build_supervised_example(
    body: str,
    label: str,
    condition: str,
    stoi: Dict[str, int],
    fillers: Optional[Sequence[str]] = None,
) -> SupervisedExample:
    """Build one next-token input with disjoint state and answer loss masks.

    For a target at full record index j, the corresponding model logit is at input
    index j-1.  The no-scratchpad condition and the given Step 5 versions have an
    answer target only.
    """
    expected_label = count_label(body)
    if label != expected_label:
        raise ValueError(f"wrong label {label!r} for {body!r}; expected {expected_label!r}")
    check_condition(condition)
    if (fillers is not None) != (condition in FILLER_CONDITIONS):
        raise ValueError(f"fillers are passed for the Step 5 versions only, not {condition}")

    tokens: List[str] = []
    roles: List[str] = []
    if condition == 'no_scratchpad':
        tokens.extend(body)
        roles.extend(['raw'] * len(body))
    elif condition in FILLER_CONDITIONS:
        layout_tokens, layout_roles = filler_layout(body, condition, fillers)
        tokens.extend(layout_tokens)
        roles.extend(layout_roles)
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
    expected_state_count = len(gold_state_tokens(body, condition))
    if sum(target != IGNORE_INDEX for target in state_targets) != expected_state_count:
        raise AssertionError("unexpected number of scratchpad targets")

    return SupervisedExample(
        input_ids=input_ids,
        state_targets=state_targets,
        answer_targets=answer_targets,
        record=''.join(tokens),
    )
