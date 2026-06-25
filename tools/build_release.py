from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from cannbench.release import build_release_archive


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("dist"))
    parser.add_argument("--release-name", default="cannbench-release")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    repo_root = REPO_ROOT
    stage_dir = args.output_dir / args.release_name
    archive_path = args.output_dir / f"{args.release_name}.tar.gz"
    build_release_archive(
        repo_root=repo_root,
        output_path=archive_path,
        stage_dir=stage_dir,
        dtype=args.dtype,
        seed=args.seed,
    )
    print(archive_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
