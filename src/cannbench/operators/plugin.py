from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from cannbench.core.result import OperatorCase
from cannbench.operators.spec import OperatorSpec


@dataclass(frozen=True)
class TorchOperatorContext:
    backend: Any
    torch: Any
    request: Any
    case: Any
    device: Any
    dtype: Any


@dataclass(frozen=True)
class OperatorPlugin:
    spec: OperatorSpec
    get_dataset: Callable[[str], Any]
    get_case: Callable[[str, str], Any]
    materialize_inputs: Callable[..., dict[str, object]]
    build_torch_callable: Callable[[TorchOperatorContext], Callable[[], Any]]
    sort_order: int = 1000

    def build_result_case(self, case: Any) -> OperatorCase:
        return OperatorCase(
            case_id=case.case_id,
            family=case.family,
            source_kind=case.source_kind,
            source_project=case.source_project,
            source_model=case.source_model,
            source_file=case.source_file,
            source_op=case.source_op,
            payload=case.payload,
        )
