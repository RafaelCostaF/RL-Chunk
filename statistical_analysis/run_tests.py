#!/usr/bin/env python3
"""Testes estatisticos pos-hoc (Friedman + Nemenyi CD, Friedman + Bonferroni CD,
paired t-test todos-os-pares) sobre os 11 metodos de table.tex, por metrica.

Granularidade: por pergunta, pareado pela intersecao de interaction_id entre os
metodos comparados nesse cenario (decisao confirmada com o usuario).

Dois cenarios:
  - with_gemini: so as 4 metricas de texto (ROUGE-L F1, BLEU-4, BERTScore F1,
    Cosine Similarity), 11 metodos - RAGAS nao existe para o Gemini.
  - without_gemini: as 9 metricas de qualidade (texto + RAGAS), 10 metodos.
Mean Chunks fica de fora dos testes (nao e metrica de qualidade de resposta, e
o Gemini usa o texto inteiro em vez de um orcamento de chunks - nao comparavel
na mesma escala).

Uso: python run_tests.py
"""

from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import scipy.stats as stats
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUTPUT = Path(__file__).parent / "output"

TEXT_METRICS = ["ROUGE-L F1", "BLEU-4", "BERTScore F1", "Cosine Similarity"]
RAGAS_METRICS = ["Faithfulness", "Answer Relevancy", "Answer Correctness", "Context Precision", "Context Recall"]
ALL_QUALITY_METRICS = TEXT_METRICS + RAGAS_METRICS

METHOD_COLORS = {
    "PPO": "#2a78d6",
    "DDPG": "#eb6834",
    "R-PPO": "#1baf7a",
    "SAC": "#eda100",
    "Search-R1 E5": "#e87ba4",
    "FAISS": "#008300",
    "BM25": "#4a3aa7",
    "ColBERT": "#e34948",
    "Deep Retrieval+FAISS": "#8c564b",
    "Deep Retrieval+BM25": "#7f7f7f",
    "Gemini": "#1a1a1a",
}

ALPHA = 0.05


def wide_matrix(df: pd.DataFrame, metric: str, methods: list[str]) -> pd.DataFrame:
    sub = df[(df["metric"] == metric) & (df["method"].isin(methods))]
    wide = sub.pivot_table(index="interaction_id", columns="method", values="value")
    return wide[methods].dropna()


def bonferroni_dunn_matrix(wide: pd.DataFrame) -> pd.DataFrame:
    """Matriz de p-valores pareados (Wilcoxon signed-rank, o teste nao-parametrico
    pareado padrao) com correcao de Bonferroni para todos os pares - a mesma base
    (postos de Friedman) do Nemenyi, mas com uma correcao de multiplicidade mais
    conservadora (Demsar 2006)."""
    methods = list(wide.columns)
    n = len(methods)
    pmat = pd.DataFrame(np.ones((n, n)), index=methods, columns=methods)
    pvals, pairs = [], []
    for a, b in itertools.combinations(methods, 2):
        try:
            _, p = stats.wilcoxon(wide[a], wide[b])
        except ValueError:
            # todas as diferencas sao zero (os dois metodos deram o mesmo valor
            # em toda pergunta pareada) - sem evidencia de diferenca, p=1.0
            p = 1.0
        pvals.append(p)
        pairs.append((a, b))
    _, p_adj, _, _ = multipletests(pvals, alpha=ALPHA, method="bonferroni")
    for (a, b), p in zip(pairs, p_adj):
        pmat.loc[a, b] = pmat.loc[b, a] = min(p, 1.0)
    return pmat


def paired_ttest_matrix(wide: pd.DataFrame) -> pd.DataFrame:
    methods = list(wide.columns)
    n = len(methods)
    pmat = pd.DataFrame(np.ones((n, n)), index=methods, columns=methods)
    tmat = pd.DataFrame(np.zeros((n, n)), index=methods, columns=methods)
    for a, b in itertools.combinations(methods, 2):
        t, p = stats.ttest_rel(wide[a], wide[b])
        if np.isnan(p):  # variancia zero da diferenca (valores identicos par a par)
            t, p = 0.0, 1.0
        pmat.loc[a, b] = pmat.loc[b, a] = p
        tmat.loc[a, b] = t
        tmat.loc[b, a] = -t
    return pmat, tmat


def plot_cd_diagram(ranks: pd.Series, sig_matrix: pd.DataFrame, title: str, out_path: Path):
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(ranks) + 1.5))
    palette = {m: METHOD_COLORS.get(m, "#888888") for m in ranks.index}
    sp.critical_difference_diagram(ranks, sig_matrix, ax=ax, color_palette=palette)
    ax.set_title(title, fontsize=12, color="#0b0b0b", pad=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def paired_heatmap(pmat: pd.DataFrame, title: str, out_path: Path, annotate: pd.DataFrame | None = None):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    order = pmat.index
    data = pmat.loc[order, order].values
    im = ax.imshow(data, cmap="RdYlGn_r", vmin=0, vmax=0.2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8)
    for i in range(len(order)):
        for j in range(len(order)):
            if i == j:
                continue
            v = data[i, j]
            txt = f"{v:.3f}" + ("*" if v < ALPHA else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=6, color="black")
    ax.set_title(title, fontsize=12, pad=12)
    fig.colorbar(im, ax=ax, label="p-value", shrink=0.8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)


def run_metric(df: pd.DataFrame, metric: str, methods: list[str], out_dir: Path):
    wide = wide_matrix(df, metric, methods)
    n, k = wide.shape
    if n < 10:
        print(f"  [skip] {metric}: so {n} perguntas pareadas (intersecao pequena demais)")
        return None
    out_dir.mkdir(parents=True, exist_ok=True)

    fried_stat, fried_p = stats.friedmanchisquare(*[wide[c].values for c in wide.columns])
    ranks = wide.rank(axis=1, ascending=False).mean().sort_values()  # rank 1 = melhor

    # --- Nemenyi ---
    nem = sp.posthoc_nemenyi_friedman(wide.values)
    nem.columns = wide.columns
    nem.index = wide.columns
    nem = nem.loc[ranks.index, ranks.index]
    plot_cd_diagram(
        ranks, nem, f"{metric} - CD diagram (Nemenyi, N={n})", out_dir / "cd_nemenyi.png"
    )
    nem.to_csv(out_dir / "nemenyi_pvalues.csv")

    # --- Bonferroni-Dunn (Wilcoxon pareado, Bonferroni) ---
    bonf = bonferroni_dunn_matrix(wide).loc[ranks.index, ranks.index]
    plot_cd_diagram(
        ranks, bonf, f"{metric} - CD diagram (Bonferroni, N={n})", out_dir / "cd_bonferroni.png"
    )
    bonf.to_csv(out_dir / "bonferroni_pvalues.csv")

    # --- paired t-test, todos os pares ---
    ttest_p, ttest_t = paired_ttest_matrix(wide)
    ttest_p = ttest_p.loc[ranks.index, ranks.index]
    ttest_t = ttest_t.loc[ranks.index, ranks.index]
    ttest_p.to_csv(out_dir / "paired_ttest_pvalues.csv")
    ttest_t.to_csv(out_dir / "paired_ttest_tstats.csv")
    paired_heatmap(ttest_p, f"{metric} - paired t-test p-values (N={n})", out_dir / "paired_ttest_heatmap.png")

    long_pairs = []
    for a, b in itertools.combinations(ranks.index, 2):
        long_pairs.append({
            "method_a": a, "method_b": b,
            "mean_a": wide[a].mean(), "mean_b": wide[b].mean(),
            "t_stat": ttest_t.loc[a, b], "p_value": ttest_p.loc[a, b],
            "significant_p05": ttest_p.loc[a, b] < ALPHA,
        })
    pd.DataFrame(long_pairs).to_csv(out_dir / "paired_ttest_table.csv", index=False)

    summary = {
        "metric": metric, "n_paired": n, "n_methods": k,
        "friedman_stat": fried_stat, "friedman_p": fried_p,
        "friedman_significant": fried_p < ALPHA,
        "best_method": ranks.index[0], "best_avg_rank": ranks.iloc[0],
        "worst_method": ranks.index[-1], "worst_avg_rank": ranks.iloc[-1],
    }
    ranks.to_csv(out_dir / "avg_ranks.csv", header=["avg_rank"])
    return summary


def run_scenario(df: pd.DataFrame, name: str, metrics: list[str], methods: list[str]):
    print(f"\n=== Cenario: {name} ({len(methods)} metodos: {', '.join(methods)}) ===")
    scenario_dir = OUTPUT / name
    summaries = []
    for metric in metrics:
        print(f"-- {metric} --")
        out_dir = scenario_dir / metric.lower().replace(" ", "_")
        s = run_metric(df, metric, methods, out_dir)
        if s:
            summaries.append(s)
    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(scenario_dir / "summary.csv", index=False)
    print(f"\nResumo ({name}):")
    print(summary_df.to_string(index=False))
    return summary_df


def main():
    df = pd.read_parquet(OUTPUT / "long_format_all_methods.parquet")

    all_methods = sorted(df["method"].unique())
    methods_without_gemini = [m for m in all_methods if m != "Gemini"]
    methods_with_gemini = all_methods

    run_scenario(df, "with_gemini", TEXT_METRICS, methods_with_gemini)
    run_scenario(df, "without_gemini", ALL_QUALITY_METRICS, methods_without_gemini)


if __name__ == "__main__":
    main()
