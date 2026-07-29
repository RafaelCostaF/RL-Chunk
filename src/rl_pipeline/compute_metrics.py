#!/usr/bin/env python3
"""Computes the final per-row metrics (ROUGE-L F1, BLEU-4, BERTScore F1,
Cosine Similarity, Faithfulness, Answer Relevancy, Answer Correctness,
Context Precision, Context Recall) and assembles the final domain x
algorithm table, with mean_chunks_selected.

ROUGE-L F1 / BLEU-4 / BERTScore F1 / Cosine Similarity use the same
functions/formulas as the old pipeline (6-metricas-finais.ipynb cell 60),
just applied to the already-cleaned response text (the old pipeline had a
bug where the text used for these 4 metrics sometimes still contained the
raw tuple "(text, tokens_in, tokens_out)" instead of plain text - see the
improvement points from earlier discussion). As in the old pipeline, these 4
metrics are computed only over rows with a non-empty llm_response (same
logic as urgente_testando_treinamentos-rl.ipynb cell 24); RAGAS and
mean_chunks_selected use all rows.

Usage:
    python compute_metrics.py --input responses.parquet --output-dir results/
"""

import argparse
import ast
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bert_score
import numpy as np
import pandas as pd
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge_score import rouge_scorer
from scipy.spatial.distance import cosine

import llm_cache  # noqa: F401 - side effect: sets up the local cache (see llm_cache.py)
from config import ALGO_LABELS, ALGO_ORDER, DOMAIN_ORDER, METRIC_ORDER, OPENAI_API_KEY, OPENAI_MODEL
from env import get_nlp


def compute_bleu(reference: str, hypothesis: str) -> float:
    reference_tokens = reference.split()
    hypothesis_tokens = hypothesis.split()
    smoothie = SmoothingFunction().method4
    return sentence_bleu(
        [reference_tokens], hypothesis_tokens,
        weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie,
    )


def compute_rouge_l(reference: str, hypothesis: str) -> float:
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return scorer.score(reference, hypothesis)["rougeL"].fmeasure


def compute_cosine_similarity(reference: str, hypothesis: str) -> float:
    nlp = get_nlp()
    doc1, doc2 = nlp(reference), nlp(hypothesis)
    if doc1.vector_norm and doc2.vector_norm:
        return 1 - cosine(doc1.vector, doc2.vector)
    return 0.0


def as_list(x):
    if isinstance(x, (list, np.ndarray)):
        return list(x)
    try:
        return ast.literal_eval(x)
    except Exception:
        return [str(x)]


def n_chunks(x) -> int:
    return len(as_list(x))


def compute_text_metrics(df: pd.DataFrame) -> pd.DataFrame:
    text_df = df[df["llm_response"].astype(str).str.strip() != ""].copy()
    text_df = text_df[text_df["llm_response"].notna()]

    text_cols = ["ROUGE-L F1", "BLEU-4", "Cosine Similarity", "BERTScore F1"]
    if text_df.empty:
        print(f"[warning] no rows with non-empty llm_response (out of {len(df)}) - "
              "ROUGE-L/BLEU-4/BERTScore/Cosine will be empty for this set.")
        for col in text_cols:
            text_df[col] = pd.Series(dtype=float)
        return text_df

    text_df["llm_response_lower"] = text_df["llm_response"].astype(str).str.lower()
    text_df["clean_answer_lower"] = text_df["clean_answer"].astype(str).str.lower()

    print(f"Computing ROUGE-L F1 / BLEU-4 / Cosine Similarity on {len(text_df)} rows (out of {len(df)})...")
    text_df["ROUGE-L F1"] = text_df.apply(
        lambda r: compute_rouge_l(r["clean_answer_lower"], r["llm_response_lower"]), axis=1
    )
    text_df["BLEU-4"] = text_df.apply(
        lambda r: compute_bleu(r["clean_answer_lower"], r["llm_response_lower"]), axis=1
    )
    text_df["Cosine Similarity"] = text_df.apply(
        lambda r: compute_cosine_similarity(r["clean_answer_lower"], r["llm_response_lower"]), axis=1
    )

    print("Computing BERTScore F1...")
    _, _, f1 = bert_score.score(
        text_df["llm_response_lower"].tolist(), text_df["clean_answer_lower"].tolist(),
        lang="en", verbose=False,
    )
    text_df["BERTScore F1"] = f1.numpy()
    return text_df


_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4", "gpt-5")


def _is_reasoning_model(model: str) -> bool:
    return model.lower().startswith(_REASONING_MODEL_PREFIXES)


class _IgnoreTemperatureChatOpenAI:
    """Mixin that ignores attempts to set 'temperature' - needed only for
    reasoning models (o1/o3/gpt-5*), which accept only the default
    temperature. RAGAS forces `langchain_llm.temperature = 1e-8` before every
    call (ragas/llms/base.py, LangchainLLMWrapper.generate) to try to make
    the judge deterministic - this breaks with those models (observed: a 400
    BadRequestError on every call without this mixin). "Normal" models such
    as gpt-4.1-nano (this pipeline's default) don't need this."""

    def __setattr__(self, name, value):
        if name == "temperature":
            value = None
        super().__setattr__(name, value)


def _make_evaluator_chat_openai(model: str, api_key: str):
    from langchain_openai import ChatOpenAI

    if _is_reasoning_model(model):
        cls = type("IgnoreTemperatureChatOpenAI", (_IgnoreTemperatureChatOpenAI, ChatOpenAI), {})
        return cls(model=model, api_key=api_key)
    return ChatOpenAI(model=model, api_key=api_key)


def compute_ragas_metrics(df: pd.DataFrame) -> pd.DataFrame:
    from datasets import Dataset
    from langchain_openai import OpenAIEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_correctness, answer_relevancy, context_precision, context_recall, faithfulness

    evaluator_llm = LangchainLLMWrapper(_make_evaluator_chat_openai(OPENAI_MODEL, OPENAI_API_KEY))
    evaluator_embeddings = LangchainEmbeddingsWrapper(
        llm_cache.cached_embeddings(OpenAIEmbeddings(api_key=OPENAI_API_KEY))
    )

    ragas_df = df[["clean_answer", "llm_response", "chunks_selected", "query"]].copy()
    ragas_df["chunks_selected"] = ragas_df["chunks_selected"].apply(as_list)
    ragas_df = ragas_df.rename(columns={
        "clean_answer": "answer",
        "llm_response": "generated_answer",
        "chunks_selected": "retrieved_contexts",
        "query": "user_input",
    })
    ragas_df["reference"] = ragas_df["answer"]

    dataset = Dataset.from_pandas(ragas_df)
    print(f"Computing RAGAS metrics ({OPENAI_MODEL}) on {len(ragas_df)} rows...")
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, answer_correctness, context_precision, context_recall],
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    ragas_out = result.to_pandas().reset_index(drop=True)
    ragas_out["domain"] = df["domain"].values
    ragas_out["algo"] = df["algo"].values
    return ragas_out


STD_SUFFIX = " (std. dev.)"


def build_wide_table(per_row: pd.DataFrame, text_metrics_df: pd.DataFrame, ragas_df: pd.DataFrame) -> pd.DataFrame:
    text_cols = ["ROUGE-L F1", "BLEU-4", "BERTScore F1", "Cosine Similarity"]
    ragas_cols = ["faithfulness", "answer_relevancy", "answer_correctness", "context_precision", "context_recall"]
    ragas_rename = {
        "faithfulness": "Faithfulness",
        "answer_relevancy": "Answer Relevancy",
        "answer_correctness": "Answer Correctness",
        "context_precision": "Context Precision",
        "context_recall": "Context Recall",
    }

    per_row = per_row.copy()
    per_row["n_chunks_selected"] = per_row["chunks_selected"].apply(n_chunks)

    text_grouped = text_metrics_df.groupby(["domain", "algo"])[text_cols]
    ragas_grouped = ragas_df.groupby(["domain", "algo"])[ragas_cols]
    chunks_grouped = per_row.groupby(["domain", "algo"])["n_chunks_selected"]

    def stats_table(agg: str) -> pd.DataFrame:
        table = text_grouped.agg(agg).join(getattr(ragas_grouped, agg)().rename(columns=ragas_rename), how="outer")
        table["mean_chunks_selected"] = getattr(chunks_grouped, agg)()
        return table

    def to_wide(df: pd.DataFrame, suffix: str = "") -> pd.DataFrame:
        w = df.stack().unstack("algo")
        w.index.set_names(["domain", "metric"], inplace=True)
        if suffix:
            w = w.rename(index=lambda m: f"{m}{suffix}", level="metric")
        present_algos = [a for a in ALGO_ORDER if a in w.columns]
        return w.reindex(present_algos, axis=1).rename(columns=ALGO_LABELS)

    wide_mean = to_wide(stats_table("mean"))
    wide_std = to_wide(stats_table("std"), suffix=STD_SUFFIX)

    present_domains = [d for d in DOMAIN_ORDER if d in per_row["domain"].unique()]
    metric_row_order = [label for m in METRIC_ORDER for label in (m, f"{m}{STD_SUFFIX}")]

    wide = pd.concat([wide_mean, wide_std])
    wide = wide.reindex(pd.MultiIndex.from_product([present_domains, metric_row_order], names=["domain", "metric"]))
    return wide


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="parquet generated by generate_responses.py")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    df = pd.read_parquet(args.input).reset_index(drop=True)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    text_metrics_df = compute_text_metrics(df)
    ragas_df = compute_ragas_metrics(df)

    text_metrics_df.to_parquet(output_dir / "text_metrics_per_row.parquet")
    ragas_df.to_parquet(output_dir / "ragas_metrics_per_row.parquet")

    wide = build_wide_table(df, text_metrics_df, ragas_df)
    wide.to_csv(output_dir / "final_rl_proposal_table.csv", float_format="%.6f")

    display_df = wide.reset_index()
    display_df["domain"] = display_df["domain"].mask(display_df["domain"].duplicated(), "")
    with pd.option_context("display.max_rows", None, "display.width", 160):
        print("\nQuality evaluation of generated answers - Proposal: RL-based chunk selection\n")
        print(display_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    try:
        with pd.ExcelWriter(output_dir / "final_rl_proposal_table.xlsx", engine="openpyxl") as writer:
            wide.to_excel(writer, merge_cells=True, float_format="%.4f")
    except ImportError:
        print("openpyxl not installed - skipping .xlsx export (CSV was already saved).")

    print(f"\nResults saved to {output_dir}/")


if __name__ == "__main__":
    main()
