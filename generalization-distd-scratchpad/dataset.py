"""Load the shared raw data and build condition-specific teacher-forced tensors."""
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from scratchpad import build_supervised_example


@dataclass
class TensorSplit:
    inputs: torch.Tensor
    state_targets: torch.Tensor
    answer_targets: torch.Tensor
    bodies: List[str]
    labels: List[str]


def load_meta(data_dir: str) -> Dict:
    with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as file:
        return pickle.load(file)


def load_raw_examples(data_dir: str, name: str, meta: Dict) -> List[Tuple[str, str]]:
    ids = np.fromfile(os.path.join(data_dir, f'{name}.bin'), dtype=np.uint16)
    decoded = ''.join(meta['itos'][int(token_id)] for token_id in ids)
    lines = [line for line in decoded.splitlines() if line]
    examples = []
    for line in lines:
        body, label = line.split(':')
        if len(body) != meta['length']:
            raise ValueError(f"{name} example has raw length {len(body)}: {line!r}")
        examples.append((body, label))
    return examples


def tensorize_examples(examples, condition: str, meta: Dict) -> TensorSplit:
    built = [
        build_supervised_example(
            body, label, condition, meta['stoi'], threshold=meta['threshold']
        )
        for body, label in examples
    ]
    lengths = {len(example.input_ids) for example in built}
    if len(lengths) != 1:
        raise ValueError(f"condition {condition!r} produced mixed sequence lengths: {lengths}")
    return TensorSplit(
        inputs=torch.tensor([example.input_ids for example in built], dtype=torch.long),
        state_targets=torch.tensor(
            [example.state_targets for example in built], dtype=torch.long
        ),
        answer_targets=torch.tensor(
            [example.answer_targets for example in built], dtype=torch.long
        ),
        bodies=[body for body, _ in examples],
        labels=[label for _, label in examples],
    )


def load_tensor_split(data_dir: str, name: str, condition: str, meta: Dict) -> TensorSplit:
    return tensorize_examples(load_raw_examples(data_dir, name, meta), condition, meta)
