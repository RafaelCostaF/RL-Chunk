"""Local cache (SQLite) for LLM and embeddings calls.

Purpose: when reprocessing (retraining and running inference again, or just
re-running after an error/credit shortage), if the sent PROMPT is identical
to a previous call (same query + same chunks, since both go into the prompt
text), the answer is read from the local database instead of calling the API
again. This applies both to this pipeline's direct calls (llm.py) and to
RAGAS's calls (compute_metrics.py), because both go through langchain's
ChatOpenAI, which automatically respects this global cache.

Just import this module (the import's side effect already sets up the
cache) before any LLM call.
"""

from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from config import PIPELINE_DIR

CACHE_DIR = PIPELINE_DIR / ".llm_cache"
CACHE_DIR.mkdir(exist_ok=True)

set_llm_cache(SQLiteCache(database_path=str(CACHE_DIR / "chat_cache.sqlite")))


def cached_embeddings(embeddings):
    """Wraps a langchain embeddings object (e.g., OpenAIEmbeddings) with an
    on-disk cache (key = text + model) - used by RAGAS (answer_relevancy)."""
    from langchain.embeddings import CacheBackedEmbeddings
    from langchain.storage import LocalFileStore

    store = LocalFileStore(str(CACHE_DIR / "embeddings"))
    return CacheBackedEmbeddings.from_bytes_store(
        embeddings, store, namespace=getattr(embeddings, "model", "openai-embeddings"),
        query_embedding_cache=True,  # defaults to caching embed_documents; RAGAS also uses embed_query
    )
