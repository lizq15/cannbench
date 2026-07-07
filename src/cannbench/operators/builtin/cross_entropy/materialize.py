from __future__ import annotations

import random

from .cases import CrossEntropyCase


def materialize_cross_entropy_inputs(
    case: CrossEntropyCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    logits_size = 1
    for dim in case.logits_shape:
        logits_size *= dim
    target_size = 1
    for dim in case.target_shape:
        target_size *= dim

    logits = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(logits_size))
    targets = tuple(generator.randrange(case.num_classes) for _ in range(target_size))
    return {
        "logits_shape": case.logits_shape,
        "target_shape": case.target_shape,
        "num_classes": case.num_classes,
        "dtype": dtype,
        "logits": logits,
        "targets": targets,
    }
