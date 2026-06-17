from __future__ import annotations

import json
from pathlib import Path


def _read_perf_result(run_dir: Path) -> dict[str, object]:
    perf_dir = run_dir / "perf"
    json_files = sorted(perf_dir.glob("*.json"))
    if not json_files:
        raise ValueError(f"no perf json files found in {perf_dir}")
    return json.loads(json_files[0].read_text())


def _metric(profile: dict[str, object] | None, key: str) -> object:
    if profile is None:
        return "missing"
    return profile[key]


def _perf_row(
    result: dict[str, object], profile: dict[str, object] | None
) -> str:
    case = result["case"]
    return (
        f"| {result['backend']} | {result['device_name']} | {result['op']} | "
        f"{case['case_id']} | {result['dtype']} | "
        f"{_metric(profile, 'latency_ms_avg')} | "
        f"{_metric(profile, 'latency_ms_p50')} | "
        f"{_metric(profile, 'latency_ms_p95')} | "
        f"{_metric(profile, 'latency_ms_p99')} |"
    )


def _read_profile_summary(run_dir: Path) -> dict[str, object] | None:
    path = run_dir / "profile-summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def write_local_report(
    *,
    output_path: Path,
    nvidia_dir: Path,
    ascend_dir: Path,
    accuracy_path: Path,
) -> Path:
    nvidia = _read_perf_result(nvidia_dir)
    ascend = _read_perf_result(ascend_dir)
    nvidia_profile = _read_profile_summary(nvidia_dir)
    ascend_profile = _read_profile_summary(ascend_dir)
    accuracy = json.loads(accuracy_path.read_text())

    lines = [
        "# CannBench Local Comparison Report",
        "",
        "## Performance",
        "",
        "| backend | device_name | op | case_id | dtype | device_latency_ms_avg | device_latency_ms_p50 | device_latency_ms_p95 | device_latency_ms_p99 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        _perf_row(nvidia, nvidia_profile),
        _perf_row(ascend, ascend_profile),
        "",
        "## Accuracy",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| passed | {accuracy['passed']} |",
        f"| op | {accuracy['op']} |",
        f"| case_id | {accuracy['case_id']} |",
        f"| numel | {accuracy['numel']} |",
        f"| mismatch_count | {accuracy['mismatch_count']} |",
        f"| max_abs_error | {accuracy['max_abs_error']} |",
        f"| max_rel_error | {accuracy['max_rel_error']} |",
        f"| mean_abs_error | {accuracy['mean_abs_error']} |",
        f"| rmse | {accuracy['rmse']} |",
        f"| rtol | {accuracy['rtol']} |",
        f"| atol | {accuracy['atol']} |",
        "",
        "## Device Profile Summary",
        "",
        "| field | value |",
        "| --- | --- |",
        f"| nvidia_device_latency_ms_avg | {nvidia_profile['latency_ms_avg'] if nvidia_profile else 'missing'} |",
        f"| ascend_device_latency_ms_avg | {ascend_profile['latency_ms_avg'] if ascend_profile else 'missing'} |",
        f"| nvidia_profile_samples | {nvidia_profile['sample_count'] if nvidia_profile else 'missing'} |",
        f"| ascend_profile_samples | {ascend_profile['sample_count'] if ascend_profile else 'missing'} |",
        "",
        "## Artifacts",
        "",
        "| field | path |",
        "| --- | --- |",
        f"| nvidia_profile | {nvidia_dir / 'profile'} |",
        f"| ascend_profile | {ascend_dir / 'profile'} |",
        f"| nvidia_perf | {nvidia_dir / 'perf'} |",
        f"| ascend_perf | {ascend_dir / 'perf'} |",
        f"| accuracy | {accuracy_path} |",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    return output_path
