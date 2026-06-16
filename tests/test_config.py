import pytest

from cannbench.core.config import OperatorBenchmarkRequest


def test_operator_request_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        OperatorBenchmarkRequest(
            backend="nvidia",
            op="softmax",
            dtype="fp9",
            rows=128,
            cols=128,
            dim=-1,
            warmup=5,
            iterations=10,
        )


def test_operator_request_defaults_output_formats():
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="softmax",
        dtype="float16",
        rows=128,
        cols=128,
        dim=-1,
        warmup=5,
        iterations=10,
    )

    assert request.output_formats == ("json", "csv", "md")


@pytest.mark.parametrize("rows", [0, -1])
def test_operator_request_rejects_non_positive_rows(rows: int):
    with pytest.raises(ValueError, match="rows must be > 0"):
        OperatorBenchmarkRequest(
            backend="nvidia",
            op="softmax",
            dtype="float16",
            rows=rows,
            cols=128,
            dim=-1,
            warmup=5,
            iterations=10,
        )


@pytest.mark.parametrize("cols", [0, -1])
def test_operator_request_rejects_non_positive_cols(cols: int):
    with pytest.raises(ValueError, match="cols must be > 0"):
        OperatorBenchmarkRequest(
            backend="nvidia",
            op="softmax",
            dtype="float16",
            rows=128,
            cols=cols,
            dim=-1,
            warmup=5,
            iterations=10,
        )


@pytest.mark.parametrize("dim", [-3, 2, 3])
def test_operator_request_rejects_invalid_softmax_dim(dim: int):
    with pytest.raises(ValueError, match="dim must be one of"):
        OperatorBenchmarkRequest(
            backend="nvidia",
            op="softmax",
            dtype="float16",
            rows=128,
            cols=128,
            dim=dim,
            warmup=5,
            iterations=10,
        )
