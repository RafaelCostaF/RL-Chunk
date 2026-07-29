#!/usr/bin/env python3
"""Training/inference cost tracking for the FAISS (en_core_web_md) and BM25
baselines, using the exact same methodology as RL-Chunk's own cost table
(infer.py / train.py): codecarbon.EmissionsTracker(tracking_mode="process")
wrapping the run, plus a time.perf_counter() latency measurement per row.

Scope note (mirrors infer.py): "inference" here means chunk SELECTION only
(embedding/lexical scoring + top-k choice), not the downstream LLM answer
generation call - the same boundary RL-Chunk's own numbers use. Neither
FAISS nor BM25 has a learned/trainable component here (fixed embedder /
fixed lexical statistics), so there is no "training" phase to report for
either - only inference.

Usage:
    python cost_tracking_baselines.py --dataset ../tables/dataset_with_metrics.csv --output sustainability_baselines.csv
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker
from tqdm import tqdm

from config import MAX_SELECTED_CHUNKS
from env import chunk_text, compute_bm25_scores
from faiss_encoreweb import select_top_chunks
from env import get_nlp


def run_faiss(df: pd.DataFrame, nlp):
    tracker = EmissionsTracker(log_level="error", save_to_file=False, tracking_mode="process")
    tracker.start()
    latencies = []
    for _, row in tqdm(list(df.iterrows()), total=len(df), desc="FAISS (en_core_web_md)"):
        chunks = chunk_text(row["page_results_text"])
        t0 = time.perf_counter()
        select_top_chunks(nlp, row["query"], chunks, MAX_SELECTED_CHUNKS)
        latencies.append(time.perf_counter() - t0)
    tracker.stop()
    data = tracker.final_emissions_data
    return np.array(latencies), data


def run_bm25(df: pd.DataFrame):
    tracker = EmissionsTracker(log_level="error", save_to_file=False, tracking_mode="process")
    tracker.start()
    latencies = []
    for _, row in tqdm(list(df.iterrows()), total=len(df), desc="BM25"):
        chunks = chunk_text(row["page_results_text"])
        t0 = time.perf_counter()
        if chunks:
            scores = compute_bm25_scores(row["query"], chunks)
            order = np.argsort(-scores)[:MAX_SELECTED_CHUNKS]
            _ = [chunks[i] for i in order]
        latencies.append(time.perf_counter() - t0)
    tracker.stop()
    data = tracker.final_emissions_data
    return np.array(latencies), data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset).reset_index(drop=True)
    nlp = get_nlp()

    rows = []
    print("=== FAISS (en_core_web_md) ===")
    lat, emis = run_faiss(df, nlp)
    rows.append({
        "algo": "faiss_en_core_web_md",
        "inference_latency_s_mean": lat.mean(),
        "inference_latency_s_std": lat.std(),
        "inference_latency_s_count": len(lat),
        "emissions_kg_co2_total_run": emis.emissions if emis else np.nan,
        "energy_kwh_total_run": emis.energy_consumed if emis else np.nan,
        "wall_clock_time_s": lat.sum(),
    })
    print(f"  latency: {lat.mean():.6f} +- {lat.std():.6f} s/query")
    if emis:
        print(f"  emissions: {emis.emissions:.6e} kg CO2eq, energy: {emis.energy_consumed:.6f} kWh")

    print("\n=== BM25 ===")
    lat, emis = run_bm25(df)
    rows.append({
        "algo": "bm25",
        "inference_latency_s_mean": lat.mean(),
        "inference_latency_s_std": lat.std(),
        "inference_latency_s_count": len(lat),
        "emissions_kg_co2_total_run": emis.emissions if emis else np.nan,
        "energy_kwh_total_run": emis.energy_consumed if emis else np.nan,
        "wall_clock_time_s": lat.sum(),
    })
    print(f"  latency: {lat.mean():.6f} +- {lat.std():.6f} s/query")
    if emis:
        print(f"  emissions: {emis.emissions:.6e} kg CO2eq, energy: {emis.energy_consumed:.6f} kWh")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")
    print(out_df.round(6))


if __name__ == "__main__":
    main()
