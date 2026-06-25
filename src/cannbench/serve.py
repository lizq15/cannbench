from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


SENSITIVE_FIELDS = {
    "hostname",
    "ip",
    "username",
    "env",
    "command",
    "workdir",
    "path",
    "log",
    "stdout",
    "stderr",
    "source_code",
    "diff",
    "profile_raw",
}

RECORD_FIELDS = {
    "schema_version",
    "run_id",
    "operator",
    "dataset",
    "case_id",
    "shape",
    "dtype",
    "backend",
    "device_class",
    "implementation",
    "implementation_version",
    "metrics",
    "accuracy",
    "diff_ref",
}

METRIC_FIELDS = {"latency_ms_avg", "latency_ms_p50", "latency_ms_p95", "sample_count"}
ACCURACY_FIELDS = {"passed", "max_abs_error", "max_rel_error"}


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    accepted_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class ServeConfig:
    frontend_dir: Path
    published_dir: Path
    host: str = "127.0.0.1"
    port: int = 8000
    enable_gpu_upload: bool = False


def _is_object(value: Any) -> bool:
    return isinstance(value, dict)


def _check_sensitive_fields(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_sensitive_fields(item, f"{path}[{index}]", errors)
        return
    if not _is_object(value):
        return
    for key, child in value.items():
        if str(key).lower() in SENSITIVE_FIELDS:
            errors.append(f"sensitive field rejected at {path}.{key}")
        _check_sensitive_fields(child, f"{path}.{key}", errors)


def _reject_unknown_fields(value: dict[str, Any], allowed: set[str], path: str, errors: list[str]) -> None:
    for key in value:
        if key not in allowed:
            errors.append(f"{path}.{key} is not allowed")


def _require_string(value: Any, path: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 160:
        errors.append(f"{path} must be a non-empty string up to 160 characters")
        return None
    return value


def _require_number(value: Any, path: str, errors: list[str]) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append(f"{path} must be a non-negative finite number")
        return None
    if value < 0:
        errors.append(f"{path} must be a non-negative finite number")
        return None
    return float(value)


def _validate_record(record: Any, index: int, errors: list[str]) -> None:
    path = f"records[{index}]"
    if not _is_object(record):
        errors.append(f"{path} must be an object")
        return

    _reject_unknown_fields(record, RECORD_FIELDS, path, errors)
    if record.get("schema_version") != 1:
        errors.append(f"{path}.schema_version must be 1")

    for key in ["run_id", "operator", "dataset", "case_id", "dtype", "device_class", "implementation", "implementation_version"]:
        _require_string(record.get(key), f"{path}.{key}", errors)

    if record.get("backend") not in {"nvidia", "gpu"}:
        errors.append(f"{path}.backend must be nvidia or gpu")

    if record.get("implementation") not in {"cuda_event", "ncu"}:
        errors.append(f"{path}.implementation must be cuda_event or ncu")

    shape = record.get("shape")
    if not isinstance(shape, list) or not shape or len(shape) > 8:
        errors.append(f"{path}.shape must be a non-empty numeric array up to 8 dimensions")
    else:
        for dim_index, dimension in enumerate(shape):
            if not isinstance(dimension, int) or dimension <= 0:
                errors.append(f"{path}.shape[{dim_index}] must be a positive integer")

    metrics = record.get("metrics")
    if not _is_object(metrics):
        errors.append(f"{path}.metrics must be an object")
    else:
        _reject_unknown_fields(metrics, METRIC_FIELDS, f"{path}.metrics", errors)
        for key in METRIC_FIELDS:
            _require_number(metrics.get(key), f"{path}.metrics.{key}", errors)

    accuracy = record.get("accuracy")
    if not _is_object(accuracy):
        errors.append(f"{path}.accuracy must be an object")
    else:
        _reject_unknown_fields(accuracy, ACCURACY_FIELDS, f"{path}.accuracy", errors)
        if not isinstance(accuracy.get("passed"), bool):
            errors.append(f"{path}.accuracy.passed must be boolean")
        _require_number(accuracy.get("max_abs_error"), f"{path}.accuracy.max_abs_error", errors)
        _require_number(accuracy.get("max_rel_error"), f"{path}.accuracy.max_rel_error", errors)

    if record.get("diff_ref") is not None:
        errors.append(f"{path}.diff_ref must be null for uploaded GPU records")


def validate_gpu_benchmark_upload(payload: Any) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    _check_sensitive_fields(payload, "payload", errors)

    if not _is_object(payload):
        return ValidationResult(False, 0, ("payload must be an object",), ())
    _reject_unknown_fields(payload, {"records"}, "payload", errors)

    records = payload.get("records")
    if not isinstance(records, list):
        errors.append("payload.records must be an array")
        return ValidationResult(False, 0, tuple(errors), tuple(warnings))
    if len(records) == 0 or len(records) > 10000:
        errors.append("payload.records must contain 1 to 10000 records")

    for index, record in enumerate(records):
        _validate_record(record, index, errors)

    return ValidationResult(
        ok=not errors,
        accepted_count=len(records) if not errors else 0,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


class CannBenchRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, frontend_dir: Path, published_dir: Path, enable_gpu_upload: bool, **kwargs: Any):
        self._frontend_dir = frontend_dir
        self._published_dir = published_dir
        self._enable_gpu_upload = enable_gpu_upload
        super().__init__(*args, directory=str(frontend_dir), **kwargs)

    def translate_path(self, path: str) -> str:
        if path.startswith("/published/"):
            relative = path.removeprefix("/published/").lstrip("/")
            base = self._published_dir.resolve()
            target = (base / relative).resolve()
            try:
                target.relative_to(base)
            except ValueError:
                return str(base / "__not_found__")
            return str(target)
        return super().translate_path(path)

    def do_GET(self) -> None:
        if self.path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/gpu-results":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not self._enable_gpu_upload:
            self.send_error(HTTPStatus.FORBIDDEN, "GPU upload is disabled")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > 2 * 1024 * 1024:
            self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "payload too large")
            return

        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid JSON")
            return

        result = validate_gpu_benchmark_upload(payload)
        if not result.ok:
            self.send_error(HTTPStatus.BAD_REQUEST, "; ".join(result.errors))
            return

        uploads_dir = self._published_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        output_path = uploads_dir / f"gpu-results-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        output_path.write_text(json.dumps(payload, indent=2) + "\n")

        response = json.dumps({"ok": True, "path": str(output_path.relative_to(self._published_dir))}).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def serve_cannbench(
    *,
    frontend_dir: Path,
    published_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8000,
    enable_gpu_upload: bool = False,
) -> None:
    handler = partial(
        CannBenchRequestHandler,
        frontend_dir=frontend_dir,
        published_dir=published_dir,
        enable_gpu_upload=enable_gpu_upload,
    )
    server = ThreadingHTTPServer((host, port), handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
