from __future__ import annotations

import random

from .cases import IndexPutCase


def materialize_index_put_inputs(
    case: IndexPutCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    input_size = 1
    for dim in case.input_shape:
        input_size *= dim
    index_size = 1
    for dim in case.index_shapes[0]:
        index_size *= dim
    values_size = 1
    for dim in case.values_shape:
        values_size *= dim

    values = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(input_size))
    indices = tuple(
        tuple(generator.randrange(case.input_shape[axis]) for _ in range(index_size))
        for axis in range(len(case.index_shapes))
    )
    put_values = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(values_size))
    return {
        "input_shape": case.input_shape,
        "index_shapes": case.index_shapes,
        "values_shape": case.values_shape,
        "accumulate": case.accumulate,
        "dtype": dtype,
        "values": values,
        "indices": indices,
        "put_values": put_values,
    }
