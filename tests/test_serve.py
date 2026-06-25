from pathlib import Path

from cannbench.serve import validate_gpu_benchmark_upload


def _valid_gpu_upload():
    return {
        "records": [
            {
                "schema_version": 1,
                "run_id": "softmax-h800",
                "operator": "softmax",
                "dataset": "realistic",
                "case_id": "t5_attention",
                "shape": [4, 8, 1024, 1024],
                "dtype": "float16",
                "backend": "nvidia",
                "device_class": "H800",
                "implementation": "ncu",
                "implementation_version": "ncu",
                "metrics": {
                    "latency_ms_avg": 1.0,
                    "latency_ms_p50": 1.0,
                    "latency_ms_p95": 1.1,
                    "sample_count": 1,
                },
                "accuracy": {
                    "passed": True,
                    "max_abs_error": 0.0,
                    "max_rel_error": 0.0,
                },
                "diff_ref": None,
            }
        ]
    }


def test_validate_gpu_benchmark_upload_accepts_minimal_gpu_record():
    result = validate_gpu_benchmark_upload(_valid_gpu_upload())

    assert result.ok is True
    assert result.accepted_count == 1
    assert result.errors == ()


def test_validate_gpu_benchmark_upload_rejects_sensitive_fields():
    payload = _valid_gpu_upload()
    payload["records"][0]["env"] = {"CUDA_VISIBLE_DEVICES": "0"}

    result = validate_gpu_benchmark_upload(payload)

    assert result.ok is False
    assert "sensitive field rejected at payload.records[0].env" in result.errors


def test_validate_gpu_benchmark_upload_rejects_non_gpu_backend():
    payload = _valid_gpu_upload()
    payload["records"][0]["backend"] = "ascend"

    result = validate_gpu_benchmark_upload(payload)

    assert result.ok is False
    assert "records[0].backend must be nvidia or gpu" in result.errors
