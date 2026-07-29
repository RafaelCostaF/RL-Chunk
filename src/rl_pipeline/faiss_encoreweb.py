#!/usr/bin/env python3
"""FAISS baseline chunk selection using en_core_web_md (spaCy) instead of
all-MiniLM-L6-v2, so it uses the exact same embedding space as RL-Chunk's
reward function (env.compute_similarities) - a fair "same embedder, with vs
without RL" ablation.

Ranks chunks by similarity to the QUERY, matching RL-Chunk's reward
Align(query, chunk) (see env.py). This intentionally differs from the
original all-MiniLM-L6-v2 FAISS notebook (legacy/notebooks/6-metricas-finais
.ipynb, cell 114), which ranked chunks by similarity to the gold ANSWER -
an oracle leak not available at real inference time.

Usage:
    python faiss_encoreweb.py --dataset ../tables/dataset_with_metrics.csv --output faiss_encoreweb.parquet
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import faiss
import numpy as np
import pandas as pd
from codecarbon import EmissionsTracker
from tqdm import tqdm

from config import MAX_SELECTED_CHUNKS
from env import chunk_text, get_nlp

ALGO_NAME = "faiss_en_core_web_md"


def select_top_chunks(nlp, query: str, chunks: list[str], top_k: int) -> list[str]:
    if not chunks:
        return []

    query_doc = nlp(query)
    if not query_doc.vector_norm:
        return chunks[:top_k]

    docs = list(nlp.pipe(chunks, batch_size=200))
    vecs = np.array([d.vector for d in docs], dtype=np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vecs = vecs / norms  # normalize so inner product == cosine similarity

    q = query_doc.vector.astype(np.float32)
    q = q / (np.linalg.norm(q) or 1.0)
    q = q.reshape(1, -1)

    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    k = min(top_k, len(chunks))
    _, indices = index.search(q, k)
    return [chunks[i] for i in indices[0] if i != -1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset).reset_index(drop=True)
    nlp = get_nlp()

    tracker = EmissionsTracker(log_level="error", save_to_file=False, tracking_mode="process")
    tracker.start()

    results = []
    for idx, row in tqdm(list(df.iterrows()), total=len(df), desc="faiss (en_core_web_md)"):
        chunks = chunk_text(row["page_results_text"])
        t0 = time.perf_counter()
        selected = select_top_chunks(nlp, row["query"], chunks, MAX_SELECTED_CHUNKS)
        latency_s = time.perf_counter() - t0
        results.append({
            "algo": ALGO_NAME,
            "row_idx": idx,
            "domain": row["domain"],
            "question_type": row.get("question_type"),
            "query": row["query"],
            "answer": row["answer"],
            "page_results_text": row["page_results_text"],
            "chunks_selected": selected,
            "n_chunks_available": len(chunks),
            "inference_latency_s": latency_s,
        })

    tracker.stop()
    emissions_data = tracker.final_emissions_data
    if emissions_data:
        print(f"Emissions (kg CO2eq): {emissions_data.emissions:.6e}")
        print(f"Energy (kWh): {emissions_data.energy_consumed:.6f}")

    out_df = pd.DataFrame(results)
    out_df.to_parquet(args.output, index=False)
    print(f"Saved to {args.output} ({len(out_df)} rows)")
    print(out_df.groupby("domain")["chunks_selected"].apply(lambda s: s.map(len).mean()))
    print(out_df.groupby("domain")["inference_latency_s"].agg(["mean", "std"]))


if __name__ == "__main__":
    main()
