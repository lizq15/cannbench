from __future__ import annotations

from .cases import COMPONENT_OPERATORS, DsaPrefillCase


def materialize_dsa_prefill_inputs(
    case: DsaPrefillCase, *, dtype: str, seed: int
) -> dict[str, object]:
    return {
        "workflow": case.workflow,
        "phase": case.phase,
        "case_id": case.case_id,
        "dtype": dtype,
        "seed": seed,
        "component_ops": COMPONENT_OPERATORS,
    }
