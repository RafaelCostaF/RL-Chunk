#!/usr/bin/env python3
"""Training/inference cost tracking for Deep Retrieval + FAISS and Deep
Retrieval + BM25, using the exact same methodology as RL-Chunk's own cost
table (infer.py) and cost_tracking_baselines.py: codecarbon.EmissionsTracker
(tracking_mode="process") wrapping the run, plus time.perf_counter() latency
per row.

Scope note (explicitly requested by the user): Deep Retrieval's pipeline has
a query-rewrite step that calls the Gemini API (see
/mnt/ssd1/rafael/deep_retrieval/main.ipynb, cell 0-1) before retrieval.
codecarbon cannot measure Google's remote infrastructure energy, so that
step's cost is NOT included here - only the LOCAL retrieval step (embedding
+ FAISS search, or BM25 scoring) is timed/tracked, reusing the
already-rewritten query cached from the original run
(faiss_top10[0]['rewritten_query']) instead of calling Gemini again.

Neither variant has a learned/trainable component here (fixed embedder /
fixed lexical statistics on top of a cached, already-rewritten query), so
there is no "training" phase to report - only inference (local retrieval).

Usage:
    python cost_tracking_deep_retrieval.py \
        --dataset /mnt/ssd1/rafael/deep_retrieval/df_deep_retrieval_faiss_e_bm25.parquet \
        --output sustainability_deep_retrieval.csv
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker
from rank_bm25 import BM25Okapi
from tqdm import tqdm

TOP_K = 10
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # matches deep_retrieval/main.ipynb cell 1


def get_rewritten_query(faiss_top10) -> str | None:
    try:
        if faiss_top10 is not None and len(faiss_top10) > 0:
            first = faiss_top10[0]
            if isinstance(first, dict) and "rewritten_query" in first:
                return first["rewritten_query"]
    except Exception:
        pass
    return None


def tokenize(text) -> list[str]:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    return str(text).lower().split()


def run_faiss(df: pd.DataFrame):
    import faiss
    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODEL, device=device)
    print(f"  (SentenceTransformer device: {device})")

    tracker = EmissionsTracker(log_level="error", save_to_file=False, tracking_mode="process")
    tracker.start()
    latencies = []
    skipped = 0
    for _, row in tqdm(list(df.iterrows()), total=len(df), desc="Deep Retrieval + FAISS (local part)"):
        query = get_rewritten_query(row["faiss_top10"])
        chunks = list(row["chunks"]) if row["chunks"] is not None else []
        if not query or not chunks:
            skipped += 1
            continue
        t0 = time.perf_counter()
        chunk_vecs = model.encode(chunks, batch_size=256, show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        q_vec = model.encode([query], show_progress_bar=False, convert_to_numpy=True).astype(np.float32)
        norms = np.linalg.norm(chunk_vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        chunk_vecs = chunk_vecs / norms
        q_vec = q_vec / (np.linalg.norm(q_vec) or 1.0)
        index = faiss.IndexFlatIP(chunk_vecs.shape[1])
        index.add(chunk_vecs)
        k = min(TOP_K, len(chunks))
        index.search(q_vec, k)
        latencies.append(time.perf_counter() - t0)
    tracker.stop()
    data = tracker.final_emissions_data
    print(f"  skipped rows (no rewritten_query or no chunks): {skipped}")
    return np.array(latencies), data


def run_bm25(df: pd.DataFrame):
    tracker = EmissionsTracker(log_level="error", save_to_file=False, tracking_mode="process")
    tracker.start()
    latencies = []
    skipped = 0
    for _, row in tqdm(list(df.iterrows()), total=len(df), desc="Deep Retrieval + BM25 (local part)"):
        query = get_rewritten_query(row["faiss_top10"])
        chunks = list(row["chunks"]) if row["chunks"] is not None else []
        if not query or not chunks:
            skipped += 1
            continue
        t0 = time.perf_counter()
        bm25 = BM25Okapi([tokenize(c) for c in chunks])
        scores = np.array(bm25.get_scores(tokenize(query)))
        _ = np.argsort(-scores)[:TOP_K]
        latencies.append(time.perf_counter() - t0)
    tracker.stop()
    data = tracker.final_emissions_data
    print(f"  skipped rows (no rewritten_query or no chunks): {skipped}")
    return np.array(latencies), data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.dataset).reset_index(drop=True)
    print(f"Linhas: {len(df)}, media de chunks candidatos por linha: {df['chunks'].apply(len).mean():.1f}")

    rows = []
    print("\n=== Deep Retrieval + FAISS (local retrieval only, Gemini rewrite reused from cache) ===")
    lat, emis = run_faiss(df)
    rows.append({
        "algo": "deep_retrieval_faiss_local_only",
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

    print("\n=== Deep Retrieval + BM25 (local retrieval only, Gemini rewrite reused from cache) ===")
    lat, emis = run_bm25(df)
    rows.append({
        "algo": "deep_retrieval_bm25_local_only",
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
