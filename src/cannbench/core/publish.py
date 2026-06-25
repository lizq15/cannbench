from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class PublishResult:
    source_dir: Path
    dest_dir: Path


def publish_run_artifacts(source_dir: Path, dest_dir: Path) -> PublishResult:
    if not source_dir.is_dir():
        raise ValueError(f"source run directory does not exist: {source_dir}")

    dest_dir.mkdir(parents=True, exist_ok=True)
    for name in ("prepared", "perf", "output", "meta"):
        source_path = source_dir / name
        if source_path.is_dir():
            shutil.copytree(source_path, dest_dir / name, dirs_exist_ok=True)
    return PublishResult(source_dir=source_dir, dest_dir=dest_dir)
