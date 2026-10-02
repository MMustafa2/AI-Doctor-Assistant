"""
medical_apis.py - one function for every free API / data source of the Pakistan doctor assistant.

Install:   pip install requests python-dotenv
Run demo:  python medical_apis.py

Keys (put them in your .env file; only ICD-11 is required):
    ICD_CLIENT_ID=...        # free: register at https://icd.who.int/icdapi
    ICD_CLIENT_SECRET=...
    OPENFDA_API_KEY=...      # optional, raises the openFDA rate limit
    OPENALEX_API_KEY=...     # optional, free

Privacy rule: send ONLY medical words ("dengue", "metformin") to these APIs.
Never send a user's name, CNIC, phone number or address.
"""
import html
import os
import re
import time
import xml.etree.ElementTree as ET

import requests
from dotenv import load_dotenv

load_dotenv()
TIMEOUT = 12
HEADERS = {"User-Agent": "pk-doctor-assistant/0.1 (student project)"}


def _get(url, **kwargs):
    response = requests.get(url, timeout=TIMEOUT, headers=HEADERS, **kwargs)
    response.raise_for_status()
    return response


def _strip_tags(text):
    return html.unescape(re.sub(r"<[^>]+>", "", text or "")).strip()


# ==============================================================
# 1. WHO Global Health Observatory (no key)
# Docs: https://www.who.int/data/gho/info/gho-odata-api
# ==============================================================
def who_find_indicator(keyword, max_results=10):
    """Search WHO indicator names, e.g. 'life expectancy' -> [(code, name), ...]"""
    safe = keyword.replace("'", "''")
    r = _get("https://ghoapi.azureedge.net/api/Indicator",
             params={"$filter": f"contains(IndicatorName,'{safe}')"})
    return [(i["IndicatorCode"], i["IndicatorName"]) for i in r.json().get("value", [])[:max_results]]


def who_country_data(indicator_code, country="PAK", latest_n=5):
    """Latest values of one indicator for a country (PAK = Pakistan)."""
    r = _get(f"https://ghoapi.azureedge.net/api/{indicator_code}",
             params={"$filter": f"SpatialDim eq '{country}'"})
    rows = sorted(r.json().get("value", []), key=lambda x: x.get("TimeDim") or 0, reverse=True)
    return [{"year": x.get("TimeDim"), "value": x.get("NumericValue"), "group": x.get("Dim1")}
            for x in rows[:latest_n]]


# ==============================================================
# 2. WHO ICD-11 (free key: client id + secret)
# Docs: https://icd.who.int/icdapi
# ==============================================================
_icd_token = {"value": None, "expires": 0}


def _icd11_token():
    if _icd_token["value"] and time.time() < _icd_token["expires"]:
        return _icd_token["value"]
    client_id, secret = os.getenv("ICD_CLIENT_ID"), os.getenv("ICD_CLIENT_SECRET")
    if not client_id or not secret:
        raise RuntimeError("Set ICD_CLIENT_ID and ICD_CLIENT_SECRET in .env (register at icd.who.int/icdapi)")
    r = requests.post("https://icdaccessmanagement.who.int/connect/token", timeout=TIMEOUT, data={
        "client_id": client_id, "client_secret": secret,
        "scope": "icdapi_access", "grant_type": "client_credentials"})
    r.raise_for_status()
    data = r.json()
    _icd_token.update(value=data["access_token"], expires=time.time() + data.get("expires_in", 3600) - 60)
    return _icd_token["value"]


def icd11_search(term, max_results=5, release="2024-01"):
    """Find official disease names and ICD-11 codes. 'release' may change: see the docs."""
    r = requests.get(
        f"https://id.who.int/icd/release/11/{release}/mms/search",
        params={"q": term, "useFlexisearch": "true", "flatResults": "true"},
        headers={"Authorization": f"Bearer {_icd11_token()}", "Accept": "application/json",
                 "Accept-Language": "en", "API-Version": "v2"},
        timeout=TIMEOUT)
    r.raise_for_status()
    return [{"title": _strip_tags(e.get("title")), "code": e.get("theCode"), "id": e.get("id")}
            for e in r.json().get("destinationEntities", [])[:max_results]]


# ==============================================================
# 3. MedlinePlus (US National Library of Medicine, no key)
# Docs: https://medlineplus.gov/about/developers/webservices/
#       https://medlineplus.gov/connect/technical.html
# ==============================================================
def medlineplus_topics(term, max_results=3):
    """Keyword search of MedlinePlus health topics (returns XML, parsed here)."""
    r = _get("https://wsearch.nlm.nih.gov/ws/query",
             params={"db": "healthTopics", "term": term, "retmax": max_results})
    out = []
    for doc in ET.fromstring(r.content).iter("document"):
        fields = {c.get("name"): (c.text or "") for c in doc.findall("content")}
        out.append({"title": _strip_tags(fields.get("title")), "url": doc.get("url"),
                    "summary": _strip_tags(fields.get("FullSummary") or fields.get("snippet"))[:800]})
    return out


_CODE_SYSTEMS = {"icd10cm": "2.16.840.1.113883.6.90", "snomed": "2.16.840.1.113883.6.96",
                 "rxcui": "2.16.840.1.113883.6.88"}


def medlineplus_connect(code, system="icd10cm", lang="en"):
    """Patient-friendly info for a medical code, e.g. code='A90' (dengue), system='icd10cm'."""
    r = _get("https://connect.medlineplus.gov/service", params={
        "mainSearchCriteria.v.cs": _CODE_SYSTEMS[system], "mainSearchCriteria.v.c": code,
        "informationRecipient.languageCode.c": lang, "knowledgeResponseType": "application/json"})
    out = []
    for e in r.json().get("feed", {}).get("entry", []):
        links = e.get("link") or [{}]
        out.append({"title": (e.get("title") or {}).get("_value"), "url": links[0].get("href"),
                    "summary": _strip_tags((e.get("summary") or {}).get("_value"))[:800]})
    return out


# ==============================================================
# 4. NLM Clinical Tables (no key) - autocomplete for conditions / ICD-10-CM
# Docs: https://clinicaltables.nlm.nih.gov
# ==============================================================
def clinical_conditions(term, n=5):
    r = _get("https://clinicaltables.nlm.nih.gov/api/conditions/v3/search", params={"terms": term, "maxList": n})
    return [row[0] if isinstance(row, list) else row for row in r.json()[3]]


def clinical_icd10cm(term, n=5):
    r = _get("https://clinicaltables.nlm.nih.gov/api/icd10cm/v3/search",
             params={"sf": "code,name", "terms": term, "maxList": n})
    return [{"code": row[0], "name": row[1]} for row in r.json()[3]]


# ==============================================================
# 5. RxNorm / RxNav (no key) - drug names
# Docs: https://lhncbc.nlm.nih.gov/RxNav
# ==============================================================
# Starter list ONLY. Pakistani brands -> names that RxNorm/openFDA understand.
# Replace it with data from the DRAP registered-drug list and ask a pharmacist to review it.
PK_BRAND_TO_GENERIC = {
    "panadol": "acetaminophen", "calpol": "acetaminophen", "paracetamol": "acetaminophen",
    "brufen": "ibuprofen", "disprin": "aspirin", "flagyl": "metronidazole",
    "augmentin": "amoxicillin clavulanate", "ponstan": "mefenamic acid",
}


def to_us_generic(name):
    """US sources say 'acetaminophen', Pakistan says 'paracetamol'. Map before calling the APIs."""
    return PK_BRAND_TO_GENERIC.get(name.strip().lower(), name.strip().lower())


def rxnorm_search(name, max_results=3):
    r = _get("https://rxnav.nlm.nih.gov/REST/approximateTerm.json",
             params={"term": to_us_generic(name), "maxEntries": max_results})
    seen, out = set(), []
    for cand in r.json().get("approximateGroup", {}).get("candidate", []):
        rxcui = cand.get("rxcui")
        if rxcui and rxcui not in seen:
            seen.add(rxcui)
            props = _get(f"https://rxnav.nlm.nih.gov/REST/rxcui/{rxcui}/properties.json").json()
            out.append({"rxcui": rxcui, "name": (props.get("properties") or {}).get("name")})
    return out


# ==============================================================
# 6. openFDA drug labels (key optional)
# Docs: https://open.fda.gov/apis/
# Note: these are US labels. Do not show US doses to Pakistani users.
# ==============================================================
def openfda_label(drug_name, limit=1):
    params = {"search": f'openfda.generic_name:"{to_us_generic(drug_name)}"', "limit": limit}
    if os.getenv("OPENFDA_API_KEY"):
        params["api_key"] = os.getenv("OPENFDA_API_KEY")
    try:
        r = _get("https://api.fda.gov/drug/label.json", params=params)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 404:    # openFDA: 404 = no match
            return []
        raise
    first = lambda v: (v or [""])[0][:600]
    return [{"brand": (x.get("openfda") or {}).get("brand_name"),
             "indications": first(x.get("indications_and_usage")),
             "warnings": first(x.get("warnings"))} for x in r.json().get("results", [])]


# ==============================================================
# 7. PubMed E-utilities (no key needed for light use)
# Docs: https://www.ncbi.nlm.nih.gov/books/NBK25501/
# ==============================================================
def pubmed_search(query, n=5):
    ids = _get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",
               params={"db": "pubmed", "term": query, "retmode": "json", "retmax": n,
                       "sort": "relevance"}).json()["esearchresult"]["idlist"]
    if not ids:
        return []
    summary = _get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi",
                   params={"db": "pubmed", "id": ",".join(ids), "retmode": "json"}).json()["result"]
    return [{"title": summary[i].get("title"), "journal": summary[i].get("fulljournalname"),
             "date": summary[i].get("pubdate"), "url": f"https://pubmed.ncbi.nlm.nih.gov/{i}/"}
            for i in ids if i in summary]


# ==============================================================
# 8. OpenAlex research papers (free; a free key is recommended)
# Docs: https://developers.openalex.org
# ==============================================================
def openalex_search(query, n=5):
    params = {"search": query, "per_page": n}
    if os.getenv("OPENALEX_API_KEY"):
        params["api_key"] = os.getenv("OPENALEX_API_KEY")
    r = _get("https://api.openalex.org/works", params=params)
    return [{"title": w.get("display_name"), "year": w.get("publication_year"),
             "doi": w.get("doi"), "cited_by": w.get("cited_by_count")} for w in r.json().get("results", [])]


# ==============================================================
# 9. OpenStreetMap Overpass: hospitals, clinics, pharmacies near a point (no key)
# Docs: https://wiki.openstreetmap.org/wiki/Overpass_API   (fair use: do not hammer it)
# ==============================================================
def nearby_facilities(lat, lon, kind="hospital", radius_m=3000, n=10):
    if kind not in {"hospital", "clinic", "pharmacy", "doctors"}:
        raise ValueError("kind must be hospital, clinic, pharmacy or doctors")
    query = (f'[out:json][timeout:25];(node["amenity"="{kind}"](around:{int(radius_m)},{float(lat)},{float(lon)});'
             f'way["amenity"="{kind}"](around:{int(radius_m)},{float(lat)},{float(lon)}););out center {int(n)};')
    r = requests.post("https://overpass-api.de/api/interpreter", data={"data": query},
                      timeout=30, headers=HEADERS)
    r.raise_for_status()
    out = []
    for el in r.json().get("elements", []):
        tags = el.get("tags", {})
        centre = el.get("center", el)
        out.append({"name": tags.get("name", "(no name)"), "phone": tags.get("phone"),
                    "lat": centre.get("lat"), "lon": centre.get("lon")})
    return out


# ==============================================================
# 10. Open-Meteo weather (no key; free for non-commercial use) - heat advice
# Docs: https://open-meteo.com
# ==============================================================
def weather_now(lat, lon):
    r = _get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon, "timezone": "Asia/Karachi",
        "current": "temperature_2m,apparent_temperature,relative_humidity_2m"})
    return r.json().get("current", {})


# ==============================================================
# NO API (websites only) - link to them, do not scrape without permission:
#   DRAP registered drugs : https://eapp.dra.gov.pk/WebProductIndex.php
#   PMDC doctor search    : https://pmdc.pk/
#   NIH guidelines (PDFs) : https://www.nih.org.pk/guidelines  -> put them in knowledge_base/
# ==============================================================

if __name__ == "__main__":
    LAHORE = (31.5204, 74.3587)
    demos = [
        ("WHO: find indicator", lambda: who_find_indicator("life expectancy at birth", 3)),
        ("WHO: Pakistan data (WHOSIS_000001)", lambda: who_country_data("WHOSIS_000001")),
        ("ICD-11 search 'dengue'", lambda: icd11_search("dengue", 3)),
        ("MedlinePlus topics 'dengue'", lambda: medlineplus_topics("dengue", 2)),
        ("MedlinePlus Connect (ICD-10 A90)", lambda: medlineplus_connect("A90")),
        ("Clinical Tables conditions", lambda: clinical_conditions("dengue", 3)),
        ("Clinical Tables ICD-10-CM", lambda: clinical_icd10cm("dengue", 3)),
        ("RxNorm 'panadol'", lambda: rxnorm_search("panadol", 2)),
        ("openFDA label 'brufen'", lambda: openfda_label("brufen")),
        ("PubMed 'dengue pakistan'", lambda: pubmed_search("dengue pakistan", 3)),
        ("OpenAlex 'dengue pakistan'", lambda: openalex_search("dengue pakistan", 3)),
        ("Hospitals near Lahore centre", lambda: nearby_facilities(*LAHORE, "hospital", 3000, 3)),
        ("Weather in Lahore now", lambda: weather_now(*LAHORE)),
    ]
    for title, call in demos:
        print(f"\n=== {title}")
        try:
            print(call())
        except Exception as error:                      # one failing API must not stop the rest
            print(f"FAILED: {type(error).__name__}: {error}")