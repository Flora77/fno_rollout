#!/usr/bin/env python3
"""Freeze train/val/test MAT names and hashes before formal sparse training."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from neuralop.training.sparse_experiment_runner import (
    freeze_data_split,
    load_resolved_sparse_config,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    resolved = load_resolved_sparse_config(
        args.config, project_root=args.project_root
    )
    manifest = freeze_data_split(resolved)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest["data_root"] = os.path.relpath(
        manifest["data_root"], output.parent
    ).replace("\\", "/")
    with output.open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({"output": str(output), **manifest["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
