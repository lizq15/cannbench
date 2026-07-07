from __future__ import annotations

from cannbench.operators.builtin._dsa_fused import materialize_dsa_fused_inputs

from .cases import DsaDecodeCase


def materialize_dsa_decode_inputs(
    case: DsaDecodeCase, *, dtype: str, seed: int
) -> dict[str, object]:
    return materialize_dsa_fused_inputs(case, dtype=dtype, seed=seed)
