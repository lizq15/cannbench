from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path

SCHEMA_VERSION = 1
DATA_FILE = "output.f32"
METADATA_FILE = "output.json"


@dataclass(frozen=True)
class CapturedOperatorOutput:
    backend: str
    device_name: str
    op: str
    dtype: str
    dataset: str
    case_id: str
    seed: int
    shape: tuple[int, ...]
    values: tuple[float, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "backend": self.backend,
            "device_name": self.device_name,
            "op": self.op,
            "dtype": self.dtype,
            "dataset": self.dataset,
            "case_id": self.case_id,
            "seed": self.seed,
            "shape": list(self.shape),
            "numel": len(self.values),
            "data_file": DATA_FILE,
            "data_format": "float32_le",
        }


@dataclass(frozen=True)
class OutputComparisonResult:
    passed: bool
    shape_match: bool
    left_backend: str
    right_backend: str
    op: str
    dtype_left: str
    dtype_right: str
    case_id: str
    seed_left: int
    seed_right: int
    shape: tuple[int, ...]
    numel: int
    mismatch_count: int
    max_abs_error: float
    max_rel_error: float
    mean_abs_error: float
    rmse: float
    rtol: float
    atol: float

    def to_json_dict(self) -> dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "passed": self.passed,
            "shape_match": self.shape_match,
            "left_backend": self.left_backend,
            "right_backend": self.right_backend,
            "op": self.op,
            "dtype_left": self.dtype_left,
            "dtype_right": self.dtype_right,
            "case_id": self.case_id,
            "seed_left": self.seed_left,
            "seed_right": self.seed_right,
            "shape": list(self.shape),
            "numel": self.numel,
            "mismatch_count": self.mismatch_count,
            "max_abs_error": self.max_abs_error,
            "max_rel_error": self.max_rel_error,
            "mean_abs_error": self.mean_abs_error,
            "rmse": self.rmse,
            "rtol": self.rtol,
            "atol": self.atol,
        }


def _pack_float32_le(values: tuple[float, ...]) -> bytes:
    if not values:
        return b""
    return struct.pack(f"<{len(values)}f", *values)


def _unpack_float32_le(payload: bytes) -> tuple[float, ...]:
    if not payload:
        return ()
    count = len(payload) // 4
    return tuple(struct.unpack(f"<{count}f", payload))


def write_operator_output(
    output_dir: Path, output: CapturedOperatorOutput
) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / METADATA_FILE
    data_path = output_dir / DATA_FILE
    metadata_path.write_text(json.dumps(output.to_json_dict(), indent=2) + "\n")
    data_path.write_bytes(_pack_float32_le(output.values))
    return {"metadata": metadata_path, "data": data_path}


def read_operator_output(output_dir: Path) -> CapturedOperatorOutput:
    metadata = json.loads((output_dir / METADATA_FILE).read_text())
    if metadata.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"unsupported output schema_version: {metadata.get('schema_version')}"
        )
    values = _unpack_float32_le((output_dir / metadata["data_file"]).read_bytes())
    if len(values) != int(metadata["numel"]):
        raise ValueError(
            f"output data length mismatch: expected {metadata['numel']}, got {len(values)}"
        )
    return CapturedOperatorOutput(
        backend=metadata["backend"],
        device_name=metadata["device_name"],
        op=metadata["op"],
        dtype=metadata["dtype"],
        dataset=metadata["dataset"],
        case_id=metadata["case_id"],
        seed=int(metadata["seed"]),
        shape=tuple(int(value) for value in metadata["shape"]),
        values=values,
    )


def _round_metric(value: float) -> float:
    return round(value, 10)


def compare_operator_outputs(
    left: CapturedOperatorOutput,
    right: CapturedOperatorOutput,
    *,
    rtol: float,
    atol: float,
) -> OutputComparisonResult:
    shape_match = left.shape == right.shape
    same_length = len(left.values) == len(right.values)
    if not shape_match or not same_length:
        mismatch_count = max(len(left.values), len(right.values))
        return OutputComparisonResult(
            passed=False,
            shape_match=False,
            left_backend=left.backend,
            right_backend=right.backend,
            op=left.op,
            dtype_left=left.dtype,
            dtype_right=right.dtype,
            case_id=left.case_id,
            seed_left=left.seed,
            seed_right=right.seed,
            shape=left.shape,
            numel=max(len(left.values), len(right.values)),
            mismatch_count=mismatch_count,
            max_abs_error=math.inf,
            max_rel_error=math.inf,
            mean_abs_error=math.inf,
            rmse=math.inf,
            rtol=rtol,
            atol=atol,
        )

    abs_errors: list[float] = []
    rel_errors: list[float] = []
    mismatch_count = 0
    for left_value, right_value in zip(left.values, right.values):
        abs_error = abs(left_value - right_value)
        rel_error = abs_error / max(abs(right_value), 1e-12)
        abs_errors.append(abs_error)
        rel_errors.append(rel_error)
        if abs_error > atol + rtol * abs(right_value):
            mismatch_count += 1

    if abs_errors:
        max_abs_error = max(abs_errors)
        max_rel_error = max(rel_errors)
        mean_abs_error = sum(abs_errors) / len(abs_errors)
        rmse = math.sqrt(sum(value * value for value in abs_errors) / len(abs_errors))
    else:
        max_abs_error = max_rel_error = mean_abs_error = rmse = 0.0

    return OutputComparisonResult(
        passed=mismatch_count == 0,
        shape_match=True,
        left_backend=left.backend,
        right_backend=right.backend,
        op=left.op,
        dtype_left=left.dtype,
        dtype_right=right.dtype,
        case_id=left.case_id,
        seed_left=left.seed,
        seed_right=right.seed,
        shape=left.shape,
        numel=len(left.values),
        mismatch_count=mismatch_count,
        max_abs_error=_round_metric(max_abs_error),
        max_rel_error=_round_metric(max_rel_error),
        mean_abs_error=_round_metric(mean_abs_error),
        rmse=_round_metric(rmse),
        rtol=rtol,
        atol=atol,
    )


def write_output_comparison(path: Path, result: OutputComparisonResult) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.to_json_dict(), indent=2) + "\n")
    return path
