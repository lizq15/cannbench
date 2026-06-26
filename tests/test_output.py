from cannbench.core.output import build_benchmark_artifact_stem


def test_build_benchmark_artifact_stem_is_stable():
    assert (
        build_benchmark_artifact_stem(
            op="softmax",
            dataset="smoke",
            case_id="tiny_logits",
            dtype="float16",
            seed=7,
        )
        == "softmax-smoke-tiny_logits-float16-seed7"
    )


def test_build_benchmark_artifact_stem_includes_dataset_and_seed():
    assert (
        build_benchmark_artifact_stem(
            op="softmax",
            dataset="stress",
            case_id="wide_vocab_lm_logits",
            dtype="float16",
            seed=2,
        )
        == "softmax-stress-wide_vocab_lm_logits-float16-seed2"
    )
