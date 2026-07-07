from __future__ import annotations

import random

from .cases import TakeAlongDimCase


def materialize_take_along_dim_inputs(
    case: TakeAlongDimCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    input_size = 1
    for dim in case.input_shape:
        input_size *= dim
    index_size = 1
    for dim in case.index_shape:
        index_size *= dim

    values = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(input_size))
    indices = tuple(generator.randrange(case.input_shape[case.dim]) for _ in range(index_size))
    return {
        "input_shape": case.input_shape,
        "index_shape": case.index_shape,
        "dim": case.dim,
        "dtype": dtype,
        "values": values,
        "indices": indices,
    }
