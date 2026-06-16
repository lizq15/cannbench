import pytest

from cannbench.core.result import BenchmarkMetrics, OperatorBenchmarkResult, SoftmaxShape


def test_result_to_json_dict_contains_core_fields():
    result = OperatorBenchmarkResult(
        backend="nvidia",
        device_name="Fake GPU",
        op="softmax",
        dtype="float16",
        shape=SoftmaxShape(rows=128, cols=128, dim=-1),
        metrics=BenchmarkMetrics(
            iterations=10,
            warmup=5,
            latency_ms_avg=1.2,
            latency_ms_p50=1.1,
            latency_ms_p95=1.5,
            latency_ms_p99=1.6,
            throughput_ops_per_sec=833.33,
        ),
    )

    payload = result.to_json_dict()

    assert payload["backend"] == "nvidia"
    assert payload["metrics"]["latency_ms_avg"] == 1.2
    assert payload["shape"] == {"rows": 128, "cols": 128, "dim": -1}


@pytest.mark.parametrize("rows", [0, -1])
def test_softmax_shape_rejects_non_positive_rows(rows: int):
    with pytest.raises(ValueError, match="rows must be > 0"):
        SoftmaxShape(rows=rows, cols=128, dim=-1)


@pytest.mark.parametrize("cols", [0, -1])
def test_softmax_shape_rejects_non_positive_cols(cols: int):
    with pytest.raises(ValueError, match="cols must be > 0"):
        SoftmaxShape(rows=128, cols=cols, dim=-1)


@pytest.mark.parametrize("dim", [-3, 2, 3])
def test_softmax_shape_rejects_invalid_dim(dim: int):
    with pytest.raises(ValueError, match="dim must be one of"):
        SoftmaxShape(rows=128, cols=128, dim=dim)
