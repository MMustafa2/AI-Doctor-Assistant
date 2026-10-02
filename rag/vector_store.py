"""rag/vector_store.py - STEP 4: save / load the FAISS vector database."""
from langchain_community.vectorstores import FAISS

import config
from rag.embeddings import get_embeddings


def build_and_save(chunks):
    store = FAISS.from_documents(chunks, get_embeddings())
    config.VECTOR_DIR.parent.mkdir(parents=True, exist_ok=True)
    store.save_local(str(config.VECTOR_DIR))
    print(f"FAISS index saved to: {config.VECTOR_DIR}")
    return store


def load_store():
    if not (config.VECTOR_DIR / "index.faiss").exists():
        raise FileNotFoundError("FAISS index not found. Run first:  python build_index.py")
    return FAISS.load_local(str(config.VECTOR_DIR), get_embeddings(),
                            allow_dangerous_deserialization=True)   # our own file, safe
