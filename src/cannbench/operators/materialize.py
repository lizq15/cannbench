from __future__ import annotations

from array import array


def materialized_values_to_buffer(values: tuple[float, ...]) -> array[float]:
    return array("f", values)


def materialize_operator_inputs(
    op_name: str,
    case,
    *,
    dtype: str,
    seed: int,
) -> dict[str, object]:
    from cannbench.operators import get_operator_plugin

    return get_operator_plugin(op_name).materialize_inputs(case, dtype=dtype, seed=seed)
