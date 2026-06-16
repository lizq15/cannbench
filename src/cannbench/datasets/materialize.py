from __future__ import annotations

import random
from array import array

from cannbench.datasets.softmax import SoftmaxCase


def materialize_softmax_inputs(
    case: SoftmaxCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    size = 1
    for dim in case.shape:
        size *= dim

    values = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(size))
    return {
        "shape": case.shape,
        "dim": case.dim,
        "dtype": dtype,
        "values": values,
    }


def materialized_values_to_buffer(values: tuple[float, ...]) -> array[float]:
    return array("f", values)
