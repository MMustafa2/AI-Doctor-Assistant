"""rag/loaders.py - STEP 1: load data (web APIs from medical_apis.py + local PDF/TXT) as Documents."""
import os
from pathlib import Path

from langchain_core.documents import Document

import config
import medical_apis as api


def _safe(label, fn, *args, **kwargs):
    """One failing API must not stop the whole build."""
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        print(f"   [skip] {label}: {type(e).__name__}: {e}")
        return []


def load_disease_documents(topics):
    docs = []
    for t in topics:
        print(f"Disease: {t}")
        for item in _safe("MedlinePlus", api.medlineplus_topics, t, 2):
            if item["summary"]:
                docs.append(Document(
                    page_content=f"Health topic: {item['title']}\n{item['summary']}",
                    metadata={"source": "MedlinePlus", "url": item["url"], "topic": t}))

        codes = _safe("ICD-10-CM", api.clinical_icd10cm, t, 3)
        if codes:
            text = "\n".join(f"{c['code']}: {c['name']}" for c in codes)
            docs.append(Document(page_content=f"ICD-10-CM codes related to {t}:\n{text}",
                                 metadata={"source": "NLM Clinical Tables", "topic": t}))

        if os.getenv("ICD_CLIENT_ID"):                      # ICD-11 needs a free key
            codes11 = _safe("ICD-11", api.icd11_search, t, 3)
            if codes11:
                text = "\n".join(f"{c['code']}: {c['title']}" for c in codes11)
                docs.append(Document(page_content=f"ICD-11 codes related to {t}:\n{text}",
                                     metadata={"source": "WHO ICD-11", "topic": t}))

        for p in _safe("PubMed", api.pubmed_search, f"{t} pakistan", 3):
            docs.append(Document(
                page_content=f"Research paper about {t}: {p['title']} ({p['journal']}, {p['date']})",
                metadata={"source": "PubMed", "url": p["url"], "topic": t}))
    return docs


def load_drug_documents(drugs):
    docs = []
    for d in drugs:
        print(f"Drug: {d}")
        for lab in _safe("openFDA", api.openfda_label, d):
            text = (f"Drug: {d} (generic: {api.to_us_generic(d)})\n"
                    f"Indications: {lab['indications']}\nWarnings: {lab['warnings']}\n"
                    f"NOTE: US FDA label. Do not copy US doses for Pakistani patients.")
            docs.append(Document(page_content=text,
                                 metadata={"source": "openFDA", "topic": d}))
    return docs


def load_who_documents():
    docs = []
    for code, name in config.WHO_INDICATORS.items():
        print(f"WHO indicator: {name}")
        rows = _safe("WHO GHO", api.who_country_data, code)
        if rows:
            lines = "\n".join(f"{r['year']}: {r['value']} ({r['group']})" for r in rows)
            docs.append(Document(page_content=f"WHO data for Pakistan - {name}:\n{lines}",
                                 metadata={"source": "WHO GHO", "topic": name}))
    return docs


def load_local_files(folder=config.KNOWLEDGE_BASE_DIR):
    """PDF / TXT / MD files from knowledge_base/ (e.g. NIH Pakistan guidelines)."""
    docs = []
    for path in sorted(Path(folder).rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            from pypdf import PdfReader
            print(f"PDF: {path.name}")
            for i, page in enumerate(PdfReader(str(path)).pages):
                text = page.extract_text() or ""
                if text.strip():
                    docs.append(Document(page_content=text,
                                         metadata={"source": path.name, "page": i + 1}))
        elif suffix in {".txt", ".md"}:
            print(f"Text: {path.name}")
            docs.append(Document(page_content=path.read_text(encoding="utf-8", errors="ignore"),
                                 metadata={"source": path.name}))
    return docs


def load_all(use_api=True):
    docs = []
    if use_api:
        docs += load_disease_documents(config.DISEASE_TOPICS)
        docs += load_drug_documents(config.DRUG_TOPICS)
        docs += load_who_documents()
    docs += load_local_files()
    print(f"\nTotal documents loaded: {len(docs)}")
    return docs
