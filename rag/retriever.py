"""rag/retriever.py - STEP 5: question -> most similar chunks (this is the 'R' in RAG)."""
from functools import lru_cache

import config
from rag.vector_store import load_store


@lru_cache(maxsize=1)
def _store():
    return load_store()


def search_knowledge(query: str, k: int = config.TOP_K) -> str:
    """Return the top-k chunks as one text block with their sources, ready for the LLM."""
    docs = _store().similarity_search(query, k=k)
    if not docs:
        return "No relevant information found in the knowledge base."
    parts = []
    for i, d in enumerate(docs, 1):
        src = d.metadata.get("source", "unknown")
        extra = d.metadata.get("url") or (f"page {d.metadata['page']}" if "page" in d.metadata else "")
        parts.append(f"[{i}] (source: {src} {extra})\n{d.page_content}")
    return "\n\n".join(parts)
