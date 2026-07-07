from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files


@dataclass(frozen=True)
class MaskedSelectCase:
    case_id: str
    family: str
    input_shape: tuple[int, ...]
    mask_shape: tuple[int, ...]
    mask_density: float
    source_kind: str
    source_project: str
    source_model: str
    source_file: str
    source_op: str

    def __post_init__(self) -> None:
        input_shape = tuple(int(value) for value in self.input_shape)
        mask_shape = tuple(int(value) for value in self.mask_shape)
        if not input_shape or any(value <= 0 for value in input_shape):
            raise ValueError("input_shape must contain only positive integers")
        if not mask_shape or any(value <= 0 for value in mask_shape):
            raise ValueError("mask_shape must contain only positive integers")
        if len(mask_shape) != len(input_shape):
            raise ValueError("mask_shape must have the same rank as input_shape")
        if not 0.0 <= self.mask_density <= 1.0:
            raise ValueError("mask_density must be between 0 and 1")
        object.__setattr__(self, "input_shape", input_shape)
        object.__setattr__(self, "mask_shape", mask_shape)

    @property
    def payload(self) -> dict[str, object]:
        return {
            "input_shape": self.input_shape,
            "mask_shape": self.mask_shape,
            "mask_density": self.mask_density,
        }


@dataclass(frozen=True)
class MaskedSelectDataset:
    name: str
    cases: tuple[MaskedSelectCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))


@lru_cache(maxsize=None)
def get_masked_select_dataset(name: str) -> MaskedSelectDataset:
    resource = files(__package__).joinpath("data", f"{name}.json")
    if not resource.is_file():
        raise ValueError(f"Unknown masked_select dataset: {name}")

    payload = json.loads(resource.read_text())
    cases = tuple(
        MaskedSelectCase(
            input_shape=tuple(item["input_shape"]),
            mask_shape=tuple(item["mask_shape"]),
            **{k: v for k, v in item.items() if k not in {"input_shape", "mask_shape"}},
        )
        for item in payload["cases"]
    )
    return MaskedSelectDataset(name=payload["name"], cases=cases)


def get_masked_select_case(dataset_name: str, case_id: str) -> MaskedSelectCase:
    dataset = get_masked_select_dataset(dataset_name)
    for case in dataset.cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"Unknown masked_select case: {case_id}")
