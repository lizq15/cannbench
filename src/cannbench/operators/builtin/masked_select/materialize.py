from __future__ import annotations

import random

from .cases import MaskedSelectCase


def materialize_masked_select_inputs(
    case: MaskedSelectCase, *, dtype: str, seed: int
) -> dict[str, object]:
    generator = random.Random(seed)
    input_size = 1
    for dim in case.input_shape:
        input_size *= dim
    mask_size = 1
    for dim in case.mask_shape:
        mask_size *= dim

    values = tuple(round(generator.uniform(-1.0, 1.0), 6) for _ in range(input_size))
    mask = tuple(generator.random() < case.mask_density for _ in range(mask_size))
    return {
        "input_shape": case.input_shape,
        "mask_shape": case.mask_shape,
        "mask_density": case.mask_density,
        "dtype": dtype,
        "values": values,
        "mask": mask,
    }
