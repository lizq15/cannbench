from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactLayout:
    root: Path
    prepared_dir: Path
    perf_dir: Path
    output_dir: Path
    meta_dir: Path
    profile_dir: Path | None


def build_run_layout(base_dir: Path, run_id: str) -> ArtifactLayout:
    root = base_dir / run_id
    return ArtifactLayout(
        root=root,
        prepared_dir=root / "prepared",
        perf_dir=root / "perf",
        output_dir=root / "output",
        meta_dir=root / "meta",
        profile_dir=root / "profile",
    )


def build_published_layout(base_dir: Path, run_id: str) -> ArtifactLayout:
    root = base_dir / run_id
    return ArtifactLayout(
        root=root,
        prepared_dir=root / "prepared",
        perf_dir=root / "perf",
        output_dir=root / "output",
        meta_dir=root / "meta",
        profile_dir=None,
    )
