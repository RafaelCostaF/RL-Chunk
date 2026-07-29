# RL Chunk Selection — Pipeline

Chunk selection for RAG via RL (PPO, DDPG, Recurrent PPO, SAC): 1 model per (domain, algorithm).

## Setup

A few dataset/result files exceed GitHub's 100MB per-file limit and are
tracked as split row-chunks instead (see `.gitignore`). After cloning, run:
```bash
python3 reassemble_split_files.py
```
This reconstructs `src/tables/dataset_with_metrics.csv`,
`src/datasets/sampled_50_per_domain.parquet`, and the three
`src/rl_pipeline_run/` parquet files from their `<filename>.parts/` chunks.

OpenAI key in `src/.env` (already configured):
```
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1-nano
```

## 1. Train

```bash
src/train.sh /mnt/ssd1/rafael/rl_chunk/src/datasets/sampled_50_per_domain.parquet
```

nohup src/train.sh /mnt/ssd1/rafael/rl_chunk/src/datasets/sampled_50_per_domain.parquet > train.log 2>&1 & disown
+ 
tail -f train.log

Saves to `src/rl_pipeline_models/<domain>/<algorithm>/model.zip`. Optional: `src/train.sh <dataset> <output_dir> <timesteps>` (default 40000 timesteps).



## 2. Infer + clean answers + metrics

```bash
src/infer.sh /mnt/ssd1/rafael/rl_chunk/src/datasets/sampled_50_per_domain.parquet
```

Runs: chunk selection by the RL agent → LLM answer → cleaning → metrics (ROUGE-L F1, BLEU-4, BERTScore F1, Cosine Similarity, Faithfulness, Answer Relevancy, Answer Correctness, Context Precision, Context Recall, mean_chunks_selected).

Final result: `src/rl_pipeline_run/3_results/final_rl_proposal_table.csv` (and `.xlsx`).

For a quick/cheap test before running everything: `LIMIT=20 src/infer.sh src/datasets/sampled_50_per_domain.parquet`

## Structure

- `src/rl_pipeline/` — code (RL env, training, inference, metrics)
- `src/datasets/` — active dataset
- `statistical_analysis/` — statistical significance tests over all compared methods
- `images/` — figures used in the paper
