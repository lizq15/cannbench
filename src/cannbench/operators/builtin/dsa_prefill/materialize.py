from __future__ import annotations

from cannbench.operators.builtin._dsa_fused import materialize_dsa_fused_inputs

from .cases import DsaPrefillCase


def materialize_dsa_prefill_inputs(
    case: DsaPrefillCase, *, dtype: str, seed: int
) -> dict[str, object]:
    return materialize_dsa_fused_inputs(case, dtype=dtype, seed=seed)
