from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files

OPERATOR_NAME = "dsa_prefill"
PHASE = "prefill"
COMPONENT_OPERATORS = ("lightning_indexer", "sparse_attention")


@dataclass(frozen=True)
class DsaPrefillCase:
    case_id: str
    workflow: str
    phase: str
    family: str
    source_kind: str = "fused_workflow"
    source_project: str = "cannbench"
    source_model: str = "DeepSeek-A5-compatible"
    source_file: str = "serving_buckets/deepseek_a5_dsa.json"
    source_op: str = ""

    def __post_init__(self) -> None:
        if self.workflow != OPERATOR_NAME:
            raise ValueError(f"unsupported DSA prefill workflow: {self.workflow}")
        if self.phase != PHASE:
            raise ValueError("DSA prefill case phase mismatch")
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if not self.source_op:
            object.__setattr__(self, "source_op", self.workflow)

    @property
    def payload(self) -> dict[str, object]:
        return {
            "workflow": self.workflow,
            "phase": self.phase,
            "component_ops": COMPONENT_OPERATORS,
        }


@dataclass(frozen=True)
class DsaPrefillDataset:
    name: str
    cases: tuple[DsaPrefillCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))


@lru_cache(maxsize=None)
def get_dsa_prefill_dataset(name: str) -> DsaPrefillDataset:
    resource = files(__package__).joinpath("data", f"{name}.json")
    if not resource.is_file():
        raise ValueError(f"Unknown DSA prefill dataset: {name}")

    payload = json.loads(resource.read_text())
    cases = tuple(DsaPrefillCase(**item) for item in payload["cases"])
    return DsaPrefillDataset(name=payload["name"], cases=cases)


def get_dsa_prefill_case(dataset_name: str, case_id: str) -> DsaPrefillCase:
    dataset = get_dsa_prefill_dataset(dataset_name)
    for case in dataset.cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"Unknown DSA prefill case: {case_id}")
