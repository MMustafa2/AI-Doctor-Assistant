"""tools.py - the tools that agents can call (each tool = one API or the RAG search)."""
from langchain_core.tools import tool

import medical_apis as api
from rag.retriever import search_knowledge


@tool
def search_medical_knowledge(query: str) -> str:
    """Search the medical knowledge base (RAG over WHO, MedlinePlus, ICD, openFDA, PubMed, NIH guidelines).
    Write the query in ENGLISH medical terms, e.g. 'dengue warning signs' or 'typhoid treatment'."""
    return search_knowledge(query)


@tool
def lookup_icd_codes(disease: str) -> str:
    """Get official ICD-10-CM (and ICD-11 if configured) codes for a disease name in English."""
    lines = []
    try:
        lines += [f"ICD-10-CM {c['code']}: {c['name']}" for c in api.clinical_icd10cm(disease, 3)]
    except Exception as e:
        lines.append(f"ICD-10-CM lookup failed: {e}")
    try:
        lines += [f"ICD-11 {c['code']}: {c['title']}" for c in api.icd11_search(disease, 3)]
    except Exception:
        pass                                               # ICD-11 key may not be set
    return "\n".join(lines) or "No code found."


@tool
def search_research_papers(query: str) -> str:
    """Search PubMed for recent research papers (English query), e.g. 'dengue treatment pakistan'."""
    try:
        papers = api.pubmed_search(query, 3)
    except Exception as e:
        return f"PubMed search failed: {e}"
    return "\n".join(f"- {p['title']} ({p['journal']}, {p['date']}) {p['url']}" for p in papers) \
        or "No papers found."


@tool
def lookup_drug_info(drug_name: str) -> str:
    """Get drug facts (generic name, RxNorm id, US label indications/warnings).
    Pakistani brand names like Panadol or Brufen are converted to generic names automatically."""
    generic = api.to_us_generic(drug_name)
    out = [f"Generic name used for search: {generic}"]
    try:
        out += [f"RxNorm: {r['name']} (rxcui {r['rxcui']})" for r in api.rxnorm_search(drug_name, 1)]
    except Exception as e:
        out.append(f"RxNorm lookup failed: {e}")
    try:
        for lab in api.openfda_label(drug_name):
            out.append(f"Indications: {lab['indications']}\nWarnings: {lab['warnings']}")
    except Exception as e:
        out.append(f"openFDA lookup failed: {e}")
    out.append("NOTE: this is a US label. Do NOT copy US doses for Pakistani patients.")
    return "\n".join(out)
