"""Gymnasium environment for chunk selection.

One episode = one dataset row (query + page_results_text). The agent walks
through the row's chunks in order and decides, chunk by chunk, whether to
include it in the final selection (hard budget: MAX_SELECTED_CHUNKS).

Reward per selected chunk = quality - cost:
  - quality: a graduated bucket (graduated_reward) applied to a combination
    of cosine(query, chunk) and BM25(query, chunk) - two signals computed
    only from the query and the chunk, without using the gold answer. See
    config.REWARD_BM25_BONUS_WEIGHT and the "proxy problem" explanation
    there.
  - cost: grows exponentially with how many chunks have already been chosen
    in that episode, to incentivize selecting fewer chunks (config.
    CHUNK_COST_BASE/CHUNK_COST_GROWTH).

ChunkSelectionEnv accepts a set of rows (not just one): each reset() draws
(or sequentially walks through) a different row, allowing a single agent to
be trained that generalizes across all rows of a domain, instead of one
agent per row.
"""

from __future__ import annotations

import re

import gymnasium as gym
import numpy as np
import spacy
from gymnasium import spaces
from joblib import Memory
from rank_bm25 import BM25Okapi

from config import (
    CHUNK_COST_BASE,
    CHUNK_COST_GROWTH,
    CHUNK_MAX_LENGTH,
    MAX_SELECTED_CHUNKS,
    PIPELINE_DIR,
    REWARD_BM25_BONUS_WEIGHT,
)

_disk_cache = Memory(location=str(PIPELINE_DIR / ".spacy_cache"), verbose=0)

_NLP = None


def get_nlp():
    global _NLP
    if _NLP is None:
        _NLP = spacy.load("en_core_web_md")
        # en_core_web_md's .vector is the average of the vocabulary's static
        # (GloVe-like) vectors - it does not depend on the tagger/parser/
        # ner/tok2vec forward pass. Disabling those components makes the
        # similarity computation ~25x faster (measured: 2050 chunks in 1s vs
        # 25s with the full pipeline enabled), which matters a lot here
        # since CRAG pages can have thousands of chunks per row.
        _NLP.disable_pipes(_NLP.pipe_names)
    return _NLP


def compute_similarities(nlp, query: str, chunks: list[str]) -> np.ndarray:
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    query_doc = nlp(query)
    if not query_doc.vector_norm:
        return np.zeros(len(chunks), dtype=np.float32)
    docs = nlp.pipe(chunks, batch_size=200)
    return np.array(
        [query_doc.similarity(d) if d.vector_norm else 0.0 for d in docs], dtype=np.float32
    )


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", str(text).lower())


def compute_bm25_scores(query: str, chunks: list[str]) -> np.ndarray:
    """BM25 (rank_bm25) between the query and each chunk: a LEXICAL/sparse
    signal, complementary to cosine (semantic/dense), computed only from
    query+chunk - without using the gold answer. See the proxy problem in
    config.py: cosine of mean word vectors (spaCy) is a shallow topical
    signal that may not distinguish well between a chunk containing the
    query's exact keywords (names, numbers, dates) and a chunk that merely
    "talks about the same general subject". BM25 captures exactly that kind
    of precise lexical match. Normalized by the row's own max score (BM25
    has no fixed 0-to-1 scale like cosine)."""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    bm25 = BM25Okapi([_tokenize(c) for c in chunks])
    scores = np.array(bm25.get_scores(_tokenize(query)), dtype=np.float32)
    max_score = scores.max()
    if max_score <= 0:
        return np.zeros(len(chunks), dtype=np.float32)
    return scores / max_score


def chunk_text(text: str, max_length: int = CHUNK_MAX_LENGTH) -> list[str]:
    sentences = str(text).split(". ")
    chunks = []
    current = ""
    for sentence in sentences:
        if len(current) + len(sentence) < max_length:
            current += sentence + ". "
        else:
            if current:
                chunks.append(current.strip())
            current = sentence + ". "
    if current:
        chunks.append(current.strip())
    return chunks


@_disk_cache.cache
def chunks_and_scores_for_text(query: str, page_results_text: str):
    """Disk cache (key = query + page_results_text) so we don't recompute
    spaCy/BM25 every time the same row is trained by another algorithm, or
    revisited at inference - see PIPELINE_DIR/.spacy_cache."""
    chunks = chunk_text(page_results_text)
    query_similarities = compute_similarities(get_nlp(), query, chunks)
    bm25_scores = compute_bm25_scores(query, chunks)
    return chunks, query_similarities, bm25_scores


def combined_relevance(query_similarities, bm25_scores):
    """Semantic similarity (cosine) + additive bonus from the lexical signal
    (BM25). Additive, not a weighted average - keeps the already-working
    signal (cosine) intact for chunks with no strong lexical match, and
    gives an extra push to chunks that also match the query's exact
    keywords."""
    return query_similarities + REWARD_BM25_BONUS_WEIGHT * bm25_scores


def chunk_selection_cost(n_already_selected: int) -> float:
    return CHUNK_COST_BASE * (CHUNK_COST_GROWTH ** n_already_selected)


def graduated_reward(sim: float) -> float:
    if sim < 0.2:
        return -1.0
    if sim < 0.4:
        return -0.5
    if sim < 0.5:
        return 0.0
    if sim < 0.6:
        return 0.5
    if sim < 0.75:
        return 1.5
    if sim < 0.9:
        return 2.5
    if sim < 0.93:
        return 4.0
    if sim < 0.95:
        return 5.0
    if sim < 0.97:
        return 6.5
    if sim < 0.98:
        return 8.0
    if sim < 0.99:
        return 9.0
    return 10.0


class ChunkSelectionEnv(gym.Env):
    def __init__(self, rows, continuous_action: bool = False, row_order: str = "random", seed=None):
        super().__init__()
        self.rows = rows.reset_index(drop=True)
        if len(self.rows) == 0:
            raise ValueError("ChunkSelectionEnv received an empty set of rows")
        self.continuous_action = continuous_action
        self.row_order = row_order
        self._rng = np.random.default_rng(seed)
        self._next_row_idx = 0
        # cache of (chunks, query similarity, BM25 score) keyed by row index
        # in self.rows - a row is revisited across many episodes during
        # training, and the spaCy/BM25 computation only needs to happen once
        # per row (see chunks_and_scores_for_text).
        self._row_cache: dict[int, tuple[list[str], np.ndarray, np.ndarray]] = {}

        # similarity (cosine) and bm25_score are computed only from
        # query+chunk (never from the gold answer), so they can safely enter
        # the observation - the policy can use both signals directly in its
        # decision, not just via the reward.
        self.observation_space = spaces.Dict({
            "similarity": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "bm25_score": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "remaining_budget": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
        })
        self.action_space = (
            spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
            if continuous_action else spaces.Discrete(2)
        )

    def _pick_row_idx(self) -> int:
        if self.row_order == "random":
            return int(self._rng.integers(0, len(self.rows)))
        idx = self._next_row_idx % len(self.rows)
        self._next_row_idx += 1
        return idx

    def _chunks_and_scores(self, row_idx: int, row):
        cached = self._row_cache.get(row_idx)
        if cached is not None:
            return cached
        chunks, query_sim, bm25_scores = chunks_and_scores_for_text(row["query"], row["page_results_text"])
        self._row_cache[row_idx] = (chunks, query_sim, bm25_scores)
        return chunks, query_sim, bm25_scores

    def reset(self, *, seed=None, options=None, row=None, row_idx=None):
        super().reset(seed=seed)
        if row is not None:
            self.row = row
            self.chunks, self._similarities, self._bm25_scores = self._chunks_and_scores(
                row_idx if row_idx is not None else -1, row
            )
        else:
            idx = self._pick_row_idx()
            self.row = self.rows.iloc[idx]
            self.chunks, self._similarities, self._bm25_scores = self._chunks_and_scores(idx, self.row)
        self.query = self.row["query"]

        self.selected_chunks: list[str] = []
        self.current_idx = 0
        self.total_reward = 0.0
        self.step_count = 0
        return self._get_obs(), {}

    def step(self, action):
        truncated = False
        if self.current_idx >= len(self.chunks):
            # row with no chunks (empty page_results_text) - episode already over
            return self._get_obs(), 0.0, True, truncated, {
                "n_selected": len(self.selected_chunks), "episode": {"r": 0.0, "l": 0},
            }

        if self.continuous_action:
            action = int(np.asarray(action).reshape(-1)[0] > 0.5)

        reward = 0.0
        if action == 1:
            relevance = combined_relevance(
                self._similarities[self.current_idx], self._bm25_scores[self.current_idx]
            )
            reward = graduated_reward(float(relevance)) - chunk_selection_cost(len(self.selected_chunks))
            self.selected_chunks.append(self.chunks[self.current_idx])

        self.total_reward += reward
        self.step_count += 1
        self.current_idx += 1

        done = (
            self.current_idx >= len(self.chunks)
            or len(self.selected_chunks) >= MAX_SELECTED_CHUNKS
        )
        # "n_selected" at the top level (not inside "episode"): SB3's Monitor
        # wrapper replaces info["episode"] with its own dict (r/l/t) on done,
        # but preserves top-level keys listed in info_keywords - that's how
        # train.py logs this to tensorboard (see ChunkCountCallback).
        info = {"n_selected": len(self.selected_chunks)}
        if done:
            info["episode"] = {"r": float(self.total_reward), "l": int(self.step_count)}
        return self._get_obs(), reward, done, truncated, info

    def _get_obs(self):
        if self.current_idx >= len(self.chunks):
            return {
                "similarity": np.zeros(1, dtype=np.float32),
                "bm25_score": np.zeros(1, dtype=np.float32),
                "remaining_budget": np.zeros(1, dtype=np.float32),
            }
        remaining = MAX_SELECTED_CHUNKS - len(self.selected_chunks)
        return {
            "similarity": np.array([self._similarities[self.current_idx]], dtype=np.float32),
            "bm25_score": np.array([self._bm25_scores[self.current_idx]], dtype=np.float32),
            "remaining_budget": np.array([remaining / MAX_SELECTED_CHUNKS], dtype=np.float32),
        }


def run_episode(model, row, algo_name: str, continuous_action: bool, deterministic: bool = True,
                 is_recurrent: bool = False):
    """Runs a full episode (one row) with an already-trained policy and returns the result."""
    env = ChunkSelectionEnv(row.to_frame().T, continuous_action=continuous_action, row_order="sequential")
    obs, _ = env.reset(row=row)
    done = False
    actions_taken = []
    lstm_states = None
    episode_start = np.ones((1,), dtype=bool)
    steps = 0
    max_steps = len(env.chunks) + 1

    while not done and steps < max_steps:
        if is_recurrent:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=deterministic
            )
            episode_start = np.zeros((1,), dtype=bool)
        else:
            action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, truncated, info = env.step(action)
        actions_taken.append(int(np.asarray(action).reshape(-1)[0]))
        steps += 1

    return {
        "algo": algo_name,
        "chunks_selected": list(env.selected_chunks),
        "actions": actions_taken,
        "total_reward": env.total_reward,
        "steps": env.step_count,
        "n_chunks_available": len(env.chunks),
    }
