"""Single OpenAI client used across the pipeline (answer generation and cleaning).

Default model: gpt-4.1-nano (the same one used for RAGAS in the old
pipeline) - a "normal" (non-reasoning) model that accepts the standard
temperature/max_tokens parameters, without the workarounds gpt-5-nano
required (see earlier discussion: gpt-5-nano only accepts the default
temperature, uses max_completion_tokens, spends tokens on reasoning before
answering, and occasionally fails to return structured output for RAGAS).
To switch models, just change OPENAI_MODEL in .env - if it's a reasoning
model (o1/o3/gpt-5*), revisit _chat() and the RAGAS wrapper in
compute_metrics.py.

Uses ChatOpenAI (langchain) instead of the raw "openai" client so it
participates in the local on-disk cache configured in llm_cache.py - the
same prompt (same query + same chunks, since both go into the text) does
not call the API again.
"""

from __future__ import annotations

import llm_cache  # noqa: F401 - side effect: sets up the local cache (see llm_cache.py)
from langchain_openai import ChatOpenAI

from config import OPENAI_API_KEY, OPENAI_MODEL

_llm = None


def get_llm() -> ChatOpenAI:
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY, max_tokens=500, temperature=0.2)
    return _llm


def _chat(prompt: str, system: str) -> tuple[str, int, int]:
    response = get_llm().invoke([
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ])
    content = (response.content or "").strip()
    usage = response.usage_metadata or {}
    return content, usage.get("input_tokens", 0), usage.get("output_tokens", 0)


def get_response_from_llm(query: str, chunks) -> tuple[str, int, int]:
    """Generates the answer based exclusively on the chunks selected by the RL agent."""
    prompt = (
        "You are an intelligent assistant that answers questions exclusively based on the "
        "information provided below.\n\n"
        f"User query:\n{query}\n\n"
        f"Available sources (chunks):\n{chunks}\n\n"
        "Respond clearly, objectively, and only using the sources. Return ONLY the answer, "
        "without any additional explanations or context.\n\n"
        "If there's no answer in the sources, return an empty string.\n\n"
        "Answer:"
    )
    try:
        return _chat(prompt, "You are a helpful and concise assistant.")
    except Exception as e:
        print(f"[LLM Error] get_response_from_llm: {e}")
        return "", 0, 0


def clean_response(llm_response: str) -> str:
    """Keeps the answer if it actually answers the question based on the chunks;
    otherwise (refusal, 'I don't know', etc.) returns an empty string."""
    if not str(llm_response).strip():
        return ""
    prompt = (
        "You are an intelligent assistant reviewing a previously generated answer.\n\n"
        f"llm_response:\n{llm_response}\n\n"
        "If this response does not actually answer the question (e.g. it says the "
        "information is not available, unclear, or refuses to answer), return an empty "
        "string. Otherwise, return the llm_response exactly as it is, with no changes.\n\n"
        "Answer:"
    )
    try:
        content, _, _ = _chat(prompt, "You are a helpful and concise assistant.")
        return content
    except Exception as e:
        print(f"[LLM Error] clean_response: {e}")
        return str(llm_response)


def clean_gold_answer(query: str, answer: str) -> str:
    """Rewrites the gold-standard answer in a direct, minimal form, for a fair comparison
    with the LLM-generated answer (same logic as clean_llm_response in the old pipeline)."""
    prompt = (
        "You are a helpful assistant. Your task is to refine the following answer so that "
        "it is direct, concise, and responds precisely to the question.\n\n"
        f"Question:\n{query}\n\n"
        f"Original answer:\n{answer}\n\n"
        "Rewrite the answer to be clean, precise, and only address the question. Return "
        "ONLY the cleaned answer - no commentary, no explanations.\n\n"
        "Cleaned answer:"
    )
    try:
        content, _, _ = _chat(prompt, "You are a helpful and concise assistant.")
        return content or str(answer)
    except Exception as e:
        print(f"[LLM Error] clean_gold_answer: {e}")
        return str(answer)
