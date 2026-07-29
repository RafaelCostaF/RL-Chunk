"""Central pipeline configuration: paths, algorithms, LLM model, and API key.

The OpenAI key is read from an environment variable (OPENAI_API_KEY), loaded
from the .env file at the root of src/ (never hardcoded in the code - see
the concerns raised earlier about exposed keys in the old pipeline).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PIPELINE_DIR = Path(__file__).resolve().parent
SRC_DIR = PIPELINE_DIR.parent

load_dotenv(SRC_DIR / ".env")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4.1-nano")

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY not found. Set it in src/.env "
        "(OPENAI_API_KEY=...) or export it in the environment before running the pipeline."
    )

MAX_SELECTED_CHUNKS = 10
CHUNK_MAX_LENGTH = 500

# --- Reward: relevance-proxy weight -----------------------------------------
# The proxy problem: cosine(query, chunk) via mean word vectors (spaCy)
# measures TOPICAL relevance shallowly - it can rate a chunk "good" just for
# discussing the same general subject as the query, without containing the
# specific keywords (names, numbers, dates) the question asks for. To reduce
# this imprecision WITHOUT using the gold answer (the comparison stays
# query-vs-chunk only), the reward adds a second, complementary lexical
# signal:
#   - base: cosine(query, chunk) - semantic/dense similarity (original signal)
#   - bonus: REWARD_BM25_BONUS_WEIGHT * BM25(query, chunk) - lexical/sparse
#     similarity (rank_bm25), the same kind of signal used by this project's
#     BM25 baseline. Cosine and BM25 fail in different ways (one is semantic
#     and ignores exact term matches, the other is literal and ignores
#     synonyms/semantic paraphrase) - summing both is the classic "hybrid
#     retrieval" technique to reduce each one's blind spots.
# The bonus is ADDITIVE (it does not replace or dilute the query similarity),
# same design rationale as before: a purely semantic chunk keeps the reward
# it already had, and a chunk that also matches lexically gets an extra
# bonus.
REWARD_BM25_BONUS_WEIGHT = 0.3

# --- Reward: cost per selected chunk (incentive to select fewer) -----------
# Even with the hard budget in MAX_SELECTED_CHUNKS, we want the agent to
# prefer selecting FEWER chunks when the extra ones don't add much. Each
# included chunk subtracts a cost that GROWS exponentially with how many
# have already been selected in that episode (the 1st chunk is cheap, the
# 10th is expensive) - see the project's own notes.txt, which already
# pointed at this idea. Formula:
#   cost(n_already_selected) = CHUNK_COST_BASE * (CHUNK_COST_GROWTH ** n_already_selected)
# CALIBRATED empirically (see prior discussion) by testing on 6 and then 20
# real "finance" rows, 20-25k timesteps:
#   - 0.5/1.4 alone, with few rows (6): collapses to 0 chunks always.
#   - 0.15-0.25 (cheaper): with more rows (20), ALL algorithms go back to
#     maxing out at 10 - most chunks clear a "good enough" bar just from
#     query similarity (see REWARD_ANSWER_BONUS_WEIGHT above), so a weak
#     cost doesn't hold anyone back.
#   - 0.5/1.4 with 20 rows: SAC shows genuinely adaptive selection
#     (5-10 chunks depending on the row, not fixed) - the result we want.
#     PPO/RecurrentPPO still stick to "always 10" in this scenario; DDPG
#     stuck at "always 0" until I added ent_coef/action_noise (see train.py)
#     to keep it exploring longer - with that, DDPG converged to a stable
#     "always 7" (better than 0, but still not as per-row adaptive as SAC).
# In short: the cost mechanism works (proof: SAC), but PPO/RecurrentPPO tend
# to converge too early to the trivial "select everything" solution in these
# short tests (20-25k timesteps, few rows). With real training (full
# dataset, 40k+ timesteps by default) this may improve on its own; if
# mean_chunks_selected keeps sitting at MAX_SELECTED_CHUNKS for
# PPO/RecurrentPPO after real training, try raising ent_coef in train.py
# (0.01 -> 0.02-0.05) before touching this.
CHUNK_COST_BASE = 0.5
CHUNK_COST_GROWTH = 1.4

ALGO_ORDER = ["ppo", "ddpg", "recurrent_ppo", "sac"]
ALGO_LABELS = {
    "ppo": "PPO",
    "ddpg": "DDPG",
    "recurrent_ppo": "Recurrent PPO",
    "sac": "SAC",
}
DOMAIN_ORDER = ["finance", "movie", "open", "sports", "music"]

METRIC_ORDER = [
    "ROUGE-L F1",
    "BLEU-4",
    "BERTScore F1",
    "Cosine Similarity",
    "Faithfulness",
    "Answer Relevancy",
    "Answer Correctness",
    "Context Precision",
    "Context Recall",
    "mean_chunks_selected",
]

DEFAULT_MODELS_DIR = SRC_DIR / "rl_pipeline_models"
DEFAULT_RUN_DIR = SRC_DIR / "rl_pipeline_run"
