"""rag/chunking.py - STEP 2: split long documents into small overlapping chunks."""
from langchain_text_splitters import RecursiveCharacterTextSplitter

import config


def split_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""])
    chunks = splitter.split_documents(docs)
    print(f"Chunks created: {len(chunks)}")
    return chunks
