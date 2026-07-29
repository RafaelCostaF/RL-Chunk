#!/usr/bin/env python3
"""Runs inference: for each dataset row and each algorithm, loads the RL
model for (domain, algorithm) and runs the deterministic chunk-selection
episode.

Usage:
    python infer.py --dataset path/dataset.parquet --model-dir models/ --output inference.parquet
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker
from sb3_contrib import RecurrentPPO
from stable_baselines3 import DDPG, PPO, SAC
from tqdm import tqdm

from config import ALGO_ORDER, DEFAULT_MODELS_DIR
from env import run_episode

ALGO_CLASSES = {
    "ppo": (PPO, False),
    "recurrent_ppo": (RecurrentPPO, False),
    "ddpg": (DDPG, True),
    "sac": (SAC, True),
}

_model_cache: dict[tuple[str, str], object] = {}


def get_model(domain: str, algo_name: str, model_dir: Path):
    key = (domain, algo_name)
    if key not in _model_cache:
        algo_cls, _ = ALGO_CLASSES[algo_name]
        path = model_dir / domain / algo_name / "model.zip"
        _model_cache[key] = algo_cls.load(path) if path.exists() else None
    return _model_cache[key]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--model-dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument("--output", required=True)
    parser.add_argument("--algorithms", default=",".join(ALGO_ORDER))
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset).reset_index(drop=True)
    algos = [a.strip() for a in args.algorithms.split(",") if a.strip()]
    model_dir = Path(args.model_dir)

    missing_models = set()
    results = []

    tracker = EmissionsTracker(log_level="error", save_to_file=False, tracking_mode="process")
    tracker.start()
    inference_start = time.time()

    for idx, row in tqdm(list(df.iterrows()), total=len(df), desc="inference"):
        for algo_name in algos:
            model = get_model(row["domain"], algo_name, model_dir)
            if model is None:
                missing_models.add((row["domain"], algo_name))
                continue
            _, continuous = ALGO_CLASSES[algo_name]
            t0 = time.perf_counter()
            result = run_episode(
                model, row, algo_name, continuous,
                deterministic=True, is_recurrent=(algo_name == "recurrent_ppo"),
            )
            latency_s = time.perf_counter() - t0
            result.update({
                "row_idx": idx,
                "interaction_id": row.get("interaction_id"),
                "domain": row["domain"],
                "question_type": row.get("question_type"),
                "static_or_dynamic": row.get("static_or_dynamic"),
                "query": row["query"],
                "answer": row["answer"],
                "page_results_text": row["page_results_text"],
                "inference_latency_s": latency_s,
            })
            results.append(result)

    total_inference_time_s = time.time() - inference_start
    tracker.stop()
    emissions_data = tracker.final_emissions_data

    if missing_models:
        print("[warning] models not found (skipped):")
        for domain, algo_name in sorted(missing_models):
            print(f"   {domain}/{algo_name} - run train.py for this (domain, algorithm) first")

    out_df = pd.DataFrame(results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path)
    print(f"\nInference saved to {output_path} ({len(out_df)} rows, {out_df['algo'].nunique() if len(out_df) else 0} algorithms)")

    if not out_df.empty:
        # number of chunks selected IN THIS inference run (unrelated to
        # training - it comes from len(chunks_selected) for each episode run
        # above, deterministic, with the already-trained model).
        out_df["n_chunks_selected"] = out_df["chunks_selected"].apply(len)

        latency_by_algo = out_df.groupby("algo")[["inference_latency_s", "n_chunks_selected"]].agg(
            ["mean", "std", "count"]
        )
        latency_by_algo.columns = ["_".join(c) for c in latency_by_algo.columns]
        latency_by_algo.loc["TOTAL"] = [
            out_df["inference_latency_s"].mean(), out_df["inference_latency_s"].std(), len(out_df),
            out_df["n_chunks_selected"].mean(), out_df["n_chunks_selected"].std(), len(out_df),
        ]
        latency_by_algo["emissions_kg_co2_total_run"] = np.nan
        latency_by_algo["energy_kwh_total_run"] = np.nan
        latency_by_algo.loc["TOTAL", "emissions_kg_co2_total_run"] = emissions_data.emissions
        latency_by_algo.loc["TOTAL", "energy_kwh_total_run"] = emissions_data.energy_consumed
        latency_by_algo.loc["TOTAL", "wall_clock_time_s"] = total_inference_time_s

        latency_path = output_path.parent / "sustainability_inference.csv"
        latency_by_algo.to_csv(latency_path)
        print(f"Latency + mean_chunks_selected (mean/std per algorithm, this inference run only) + emissions saved to {latency_path}")
        print(latency_by_algo.round(4))


if __name__ == "__main__":
    main()
