#!/usr/bin/env bash
# Trains the chunk-selection RL agents: one model per (domain, algorithm)
# (PPO, DDPG, Recurrent PPO, SAC), trained over all rows of that domain in
# the dataset - generalizes to new queries within the same domain.
#
# Usage:
#   ./train.sh path/to/dataset.parquet [output_dir] [timesteps]
#
# The dataset needs the columns: domain, query, answer, page_results_text
# (same format as sampled_50_per_domain.parquet).
#
# Optional environment variables:
#   ALGORITHMS   comma-separated list (default: ppo,ddpg,recurrent_ppo,sac)
#   DOMAINS      comma-separated list (default: all domains in the dataset)
#   SEED         random seed (default: 42)
#   SKIP_EXISTING=1  skip (domain, algorithm) whose model.zip already exists

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIPELINE_DIR="$SCRIPT_DIR/rl_pipeline"

DATASET="${1:?Usage: ./train.sh path/to/dataset.parquet [output_dir] [timesteps]}"
OUTPUT_DIR="${2:-$SCRIPT_DIR/rl_pipeline_models}"
TIMESTEPS="${3:-40000}"

ALGORITHMS="${ALGORITHMS:-ppo,ddpg,recurrent_ppo,sac}"
SEED="${SEED:-42}"

EXTRA_ARGS=()
[[ -n "${DOMAINS:-}" ]] && EXTRA_ARGS+=(--domains "$DOMAINS")
[[ "${SKIP_EXISTING:-0}" == "1" ]] && EXTRA_ARGS+=(--skip-existing)

echo "== RL chunk-selection training =="
echo "Dataset:    $DATASET"
echo "Output:     $OUTPUT_DIR"
echo "Timesteps:  $TIMESTEPS (per domain x algorithm)"
echo "Algorithms: $ALGORITHMS"
echo

cd "$PIPELINE_DIR"
python3 -u train.py \
    --dataset "$DATASET" \
    --output-dir "$OUTPUT_DIR" \
    --algorithms "$ALGORITHMS" \
    --timesteps "$TIMESTEPS" \
    --seed "$SEED" \
    "${EXTRA_ARGS[@]}"

echo
echo "Models saved to: $OUTPUT_DIR/<domain>/<algorithm>/model.zip"
echo "Sustainability data (CodeCarbon + training time): $OUTPUT_DIR/sustainability_training.csv"
echo "Next step: ./infer.sh $DATASET $OUTPUT_DIR"
