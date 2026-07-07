from __future__ import annotations

from cannbench.operators.builtin._dsa_fused import (
    DsaFusedOperatorCase,
    DsaFusedOperatorDataset,
    get_dsa_fused_case,
    get_dsa_fused_dataset,
)

OPERATOR_NAME = "dsa_decode"

DsaDecodeCase = DsaFusedOperatorCase
DsaDecodeDataset = DsaFusedOperatorDataset


def get_dsa_decode_dataset(name: str) -> DsaDecodeDataset:
    return get_dsa_fused_dataset(__package__, operator_name=OPERATOR_NAME, name=name)


def get_dsa_decode_case(dataset_name: str, case_id: str) -> DsaDecodeCase:
    return get_dsa_fused_case(
        __package__,
        operator_name=OPERATOR_NAME,
        dataset_name=dataset_name,
        case_id=case_id,
    )
