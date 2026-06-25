from pathlib import Path

from cannbench.core.layout import build_published_layout, build_run_layout


def test_build_run_layout_includes_profile_dir():
    layout = build_run_layout(Path("workspace/runs"), "softmax-run")

    assert layout.root == Path("workspace/runs") / "softmax-run"
    assert layout.prepared_dir == Path("workspace/runs") / "softmax-run" / "prepared"
    assert layout.perf_dir == Path("workspace/runs") / "softmax-run" / "perf"
    assert layout.profile_dir == Path("workspace/runs") / "softmax-run" / "profile"
    assert layout.output_dir == Path("workspace/runs") / "softmax-run" / "output"
    assert layout.meta_dir == Path("workspace/runs") / "softmax-run" / "meta"


def test_build_published_layout_omits_profile_dir():
    layout = build_published_layout(Path("workspace/published"), "softmax-run")

    assert layout.root == Path("workspace/published") / "softmax-run"
    assert layout.prepared_dir == Path("workspace/published") / "softmax-run" / "prepared"
    assert layout.perf_dir == Path("workspace/published") / "softmax-run" / "perf"
    assert layout.profile_dir is None
    assert layout.output_dir == Path("workspace/published") / "softmax-run" / "output"
    assert layout.meta_dir == Path("workspace/published") / "softmax-run" / "meta"
