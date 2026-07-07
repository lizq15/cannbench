from __future__ import annotations

from cannbench.operators.builtin._dsa_fused import (
    DsaFusedOperatorCase,
    DsaFusedOperatorDataset,
    get_dsa_fused_case,
    get_dsa_fused_dataset,
)

OPERATOR_NAME = "dsa_prefill"

DsaPrefillCase = DsaFusedOperatorCase
DsaPrefillDataset = DsaFusedOperatorDataset


def get_dsa_prefill_dataset(name: str) -> DsaPrefillDataset:
    return get_dsa_fused_dataset(__package__, operator_name=OPERATOR_NAME, name=name)


def get_dsa_prefill_case(dataset_name: str, case_id: str) -> DsaPrefillCase:
    return get_dsa_fused_case(
        __package__,
        operator_name=OPERATOR_NAME,
        dataset_name=dataset_name,
        case_id=case_id,
    )
