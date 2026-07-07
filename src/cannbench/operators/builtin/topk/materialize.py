from __future__ import annotations

import random

from .cases import TopKCase


def materialize_topk_inputs(
    case: TopKCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    input_size = 1
    for dim in case.input_shape:
        input_size *= dim

    values = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(input_size))
    return {
        "input_shape": case.input_shape,
        "k": case.k,
        "dim": case.dim,
        "largest": case.largest,
        "sorted": case.sorted,
        "dtype": dtype,
        "values": values,
    }
