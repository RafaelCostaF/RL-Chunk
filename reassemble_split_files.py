#!/usr/bin/env python3
"""Reassembles the large dataset/result files that exceed GitHub's 100MB
per-file limit and were therefore split into row chunks under
"<filename>.parts/" (see .gitignore). Run this once after cloning the repo,
before using train.sh/infer.sh or the statistical analysis scripts.

Usage:
    python3 reassemble_split_files.py
"""
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent

FILES = [
    (REPO / "src/tables/dataset_with_metrics.csv", "csv"),
    (REPO / "src/rl_pipeline_run/3_results/text_metrics_per_row.parquet", "parquet"),
    (REPO / "src/rl_pipeline_run/2_responses.parquet", "parquet"),
    (REPO / "src/rl_pipeline_run/1_inference.parquet", "parquet"),
    (REPO / "src/datasets/sampled_50_per_domain.parquet", "parquet"),
]


def reassemble(path: Path, fmt: str):
    parts_dir = path.parent / f"{path.name}.parts"
    parts = sorted(parts_dir.glob(f"part_*.{fmt}"))
    if not parts:
        print(f"  [skip] no parts found in {parts_dir}")
        return
    dfs = [pd.read_csv(p) if fmt == "csv" else pd.read_parquet(p) for p in parts]
    df = pd.concat(dfs, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "csv":
        df.to_csv(path, index=False)
    else:
        df.to_parquet(path, index=False)
    print(f"  -> {path.relative_to(REPO)}: {len(df)} rows from {len(parts)} parts")


def main():
    for path, fmt in FILES:
        if path.exists():
            print(f"Skipping {path.relative_to(REPO)} (already exists)")
            continue
        print(f"Reassembling {path.relative_to(REPO)}...")
        reassemble(path, fmt)


if __name__ == "__main__":
    main()
