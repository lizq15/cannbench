from pathlib import Path

from cannbench.core.publish import publish_run_artifacts


def test_publish_run_artifacts_copies_supported_directories_only(tmp_path):
    source = tmp_path / "runs" / "softmax-run"
    dest = tmp_path / "published" / "softmax-run"
    (source / "prepared").mkdir(parents=True)
    (source / "perf").mkdir()
    (source / "output").mkdir()
    (source / "meta").mkdir()
    (source / "profile").mkdir()
    (source / "prepared" / "prepared.json").write_text("{}")
    (source / "perf" / "benchmark.json").write_text("{}")
    (source / "output" / "output.json").write_text("{}")
    (source / "meta" / "run.json").write_text("{}")
    (source / "profile" / "ncu.csv").write_text("profile")

    result = publish_run_artifacts(source, dest)

    assert result.source_dir == source
    assert result.dest_dir == dest
    assert (dest / "prepared" / "prepared.json").read_text() == "{}"
    assert (dest / "perf" / "benchmark.json").read_text() == "{}"
    assert (dest / "output" / "output.json").read_text() == "{}"
    assert (dest / "meta" / "run.json").read_text() == "{}"
    assert not (dest / "profile").exists()


def test_publish_run_artifacts_rejects_missing_source_dir(tmp_path):
    source = tmp_path / "runs" / "missing"
    dest = tmp_path / "published" / "missing"

    try:
        publish_run_artifacts(source, dest)
    except ValueError as exc:
        assert str(exc) == f"source run directory does not exist: {source}"
    else:
        raise AssertionError("expected ValueError")
