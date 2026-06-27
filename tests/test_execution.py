from pathlib import Path

from cannbench.core.execution import read_artifact_tree


def test_read_artifact_tree_returns_sorted_relative_files(tmp_path: Path):
    root = tmp_path / "artifacts"
    (root / "b").mkdir(parents=True)
    (root / "a").mkdir(parents=True)
    (root / "b" / "two.txt").write_text("two", encoding="utf-8")
    (root / "a" / "one.txt").write_text("one", encoding="utf-8")

    artifacts = read_artifact_tree(root)

    assert artifacts == (
        ("a/one.txt", b"one"),
        ("b/two.txt", b"two"),
    )
