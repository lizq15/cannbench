import json

import pytest

from cannbench.core.operator_output import (
    CapturedOperatorOutput,
    compare_operator_outputs,
    read_operator_output,
    write_operator_output,
    write_output_comparison,
)


def test_write_and_read_operator_output_artifact(tmp_path):
    output = CapturedOperatorOutput(
        backend="nvidia",
        device_name="Fake GPU",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
        shape=(2, 2),
        values=(0.1, 0.2, 0.3, 0.4),
    )

    paths = write_operator_output(tmp_path / "nvidia-output", output)
    loaded = read_operator_output(tmp_path / "nvidia-output")

    assert sorted(paths.keys()) == ["data", "metadata"]
    assert json.loads(paths["metadata"].read_text())["shape"] == [2, 2]
    assert loaded.values == pytest.approx(output.values)
    assert loaded.backend == output.backend
    assert loaded.shape == output.shape


def test_compare_operator_outputs_reports_error_metrics():
    left = CapturedOperatorOutput(
        backend="nvidia",
        device_name="Fake GPU",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
        shape=(3,),
        values=(1.0, 2.0, 3.0),
    )
    right = CapturedOperatorOutput(
        backend="ascend",
        device_name="Fake Ascend",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
        shape=(3,),
        values=(1.0, 2.01, 2.99),
    )

    result = compare_operator_outputs(left, right, rtol=1e-2, atol=1e-3)

    assert result.passed is True
    assert result.numel == 3
    assert result.mismatch_count == 0
    assert result.max_abs_error == 0.01
    assert result.left_backend == "nvidia"
    assert result.right_backend == "ascend"


def test_compare_operator_outputs_fails_on_shape_mismatch():
    left = CapturedOperatorOutput(
        backend="nvidia",
        device_name="Fake GPU",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
        shape=(2,),
        values=(1.0, 2.0),
    )
    right = CapturedOperatorOutput(
        backend="ascend",
        device_name="Fake Ascend",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
        shape=(3,),
        values=(1.0, 2.0, 3.0),
    )

    result = compare_operator_outputs(left, right, rtol=1e-3, atol=1e-3)

    assert result.passed is False
    assert result.shape_match is False
    assert result.mismatch_count == 3


def test_write_output_comparison_creates_json_report(tmp_path):
    left = CapturedOperatorOutput(
        backend="nvidia",
        device_name="Fake GPU",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
        shape=(1,),
        values=(1.0,),
    )
    right = CapturedOperatorOutput(
        backend="ascend",
        device_name="Fake Ascend",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
        shape=(1,),
        values=(1.0,),
    )
    result = compare_operator_outputs(left, right, rtol=1e-3, atol=1e-3)

    path = write_output_comparison(tmp_path / "accuracy.json", result)

    payload = json.loads(path.read_text())
    assert payload["passed"] is True
    assert payload["left_backend"] == "nvidia"
    assert payload["right_backend"] == "ascend"
