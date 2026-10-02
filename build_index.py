"""build_index.py - run ONCE (and again when you add data):  python build_index.py
Pipeline: APIs + local files -> chunking -> embeddings -> FAISS saved on disk.
Use  python build_index.py --local-only  to skip web APIs and index only knowledge_base/ files."""
import argparse
import sys

from rag.chunking import split_documents
from rag.loaders import load_all
from rag.vector_store import build_and_save


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local-only", action="store_true", help="skip web APIs")
    args = parser.parse_args()

    docs = load_all(use_api=not args.local_only)
    if not docs:
        sys.exit("No documents loaded. Check your internet connection or add files to knowledge_base/")
    build_and_save(split_documents(docs))
    print("Done. Now run:  python main.py")


if __name__ == "__main__":
    main()
