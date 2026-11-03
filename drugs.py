"""drugs.py - medicine database (data/medicines_pk.csv) + CODE safety filter.

The LLM does not invent medicines. Flow:
  diagnosis / symptoms  ->  find_candidates()  ->  safety_filter()  ->  format_rows()  ->  LLM picks from this list.
Edit the CSV in Excel to add or change medicines, brands and doses (no code change needed).
"""
import csv
import re
from functools import lru_cache
from pathlib import Path

CSV_PATH = Path(__file__).parent / "data" / "medicines_pk.csv"
MAX_ROWS = 12

AVOID_WORDS = {                      # tag in 'avoid_if' column -> words in the patient's facts that trigger it
    "kidney": ["kidney", "renal", "gurda", "gurde"],
    "liver": ["liver", "hepat", "jaundice", "cirrhosis"],
    "asthma": ["asthma", "dama"],
    "ulcer": ["ulcer", "stomach bleeding"],
}


@lru_cache(maxsize=1)
def load_medicines():
    with open(CSV_PATH, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def find_candidates(text: str):
    """Rows whose keywords appear as whole words in the text (diagnosis + complaint + symptoms).
    If one medicine matches several conditions, the most specific row (fewest keywords) is kept."""
    text = (text or "").lower()
    best = {}
    for row in load_medicines():
        keys = [k.strip() for k in row["keywords"].lower().split(";") if k.strip()]
        if any(re.search(r"\b" + re.escape(k) + r"\b", text) for k in keys):
            generic = row["generic_name"]
            if generic not in best or len(keys) < best[generic][0]:
                best[generic] = (len(keys), row)
    return [row for _, row in best.values()]


def _age(facts):
    m = re.search(r"\d+", str(facts.get("age", "")))
    return int(m.group()) if m else None


def _is_pregnant(facts):
    v = str(facts.get("pregnant", "")).lower().strip()
    return bool(v) and not re.search(r"\b(no|not|nahi|none|nope)\b", v)


def safety_filter(rows, facts, context_text=""):
    """Returns (allowed_rows, removed) where removed = [(generic_name, reason)]."""
    age = _age(facts)
    pregnant = _is_pregnant(facts)
    allergies = str(facts.get("allergies", "")).lower()
    health = (str(facts.get("chronic_diseases", "")) + " " + str(facts.get("current_medicines", ""))).lower()
    dengue = "dengue" in (context_text or "").lower()
    allowed, removed = [], []
    for row in rows:
        tags = [t.strip() for t in row["avoid_if"].split(";") if t.strip()]
        reason = None
        generic_words = [w for w in re.findall(r"[a-z]{5,}", row["generic_name"].lower())]
        if allergies and allergies != "none" and any(w in allergies for w in generic_words):
            reason = "patient allergy"
        elif "penicillin" in tags and re.search(r"penicillin|amoxicillin|augmentin|amoxil", allergies):
            reason = "penicillin allergy"
        elif "pregnancy" in tags and pregnant:
            reason = "pregnancy"
        elif "dengue" in tags and dengue:
            reason = "dengue (NSAIDs are dangerous)"
        elif "under12" in tags and age is not None and age < 12:
            reason = "child under 12"
        elif "children_only" in tags and age is not None and age >= 12:
            reason = "only for children"
        else:
            for tag, words in AVOID_WORDS.items():
                if tag in tags and any(w in health for w in words):
                    reason = f"patient has {tag} problem"
                    break
        (removed.append((row["generic_name"], reason)) if reason else allowed.append(row))
    return allowed[:MAX_ROWS], removed


def format_rows(rows):
    lines = []
    for r in rows:
        lines.append(
            f"- {r['generic_name']} ({r['composition_strength']}; formula {r['chemical_formula']}) | "
            f"Pakistani brands: {r['pk_brands']} | for: {r['target']} | "
            f"dose: {r['adult_dose']}, {r['frequency']}, duration: {r['duration']} | {r['rx_status']} | "
            f"children: {r['children_note']} | note: {r['notes']}")
    return "\n".join(lines)


# ---------------------------------------------------------------- the model's OWN suggestions (names only)
BLOCKED = {                          # never allowed as a model-only suggestion
    "antibiotic": ["amoxicillin", "ampicillin", "penicillin", "augmentin", "clavulan", "azithromycin", "clarithromycin",
                   "erythromycin", "ciprofloxacin", "levofloxacin", "ofloxacin", "metronidazole", "nitrofurantoin",
                   "cotrimoxazole", "trimethoprim", "clindamycin", "gentamicin", "meropenem", "vancomycin"],
    "steroid": ["prednisolone", "prednisone", "dexamethasone", "hydrocortisone", "betamethasone", "steroid", "cortico"],
    "controlled drug": ["tramadol", "morphine", "codeine", "pethidine", "diazepam", "alprazolam", "lorazepam",
                        "clonazepam", "zolpidem", "ketamine"],
    "painkiller (NSAID)": ["ibuprofen", "diclofenac", "naproxen", "aspirin", "mefenamic", "ketorolac", "indomethacin",
                           "piroxicam", "celecoxib", "etoricoxib", "nimesulide", "brufen", "ponstan", "disprin"],
    "blood thinner": ["warfarin", "heparin", "clopidogrel"],
}


def _words(text):
    return {w for w in re.findall(r"[a-z]{4,}", (text or "").lower())}


def _same_drug(name, row):
    return bool(_words(name) & (_words(row["generic_name"]) | _words(row["pk_brands"])))


def check_model_suggestions(suggestions, facts, allowed, removed):
    """suggestions = [{'name','reason'}]. Returns (ok, rejected). Safety is decided by code, not by the model."""
    allergies = str(facts.get("allergies", "")).lower()
    removed_words = set().union(*[_words(g) for g, _ in removed]) if removed else set()
    ok, rejected = [], []
    for item in suggestions:
        name = str(item.get("name", "")).strip()
        low = name.lower()
        if not name:
            continue
        if any(_same_drug(name, r) for r in allowed):
            continue                                       # already in the verified list
        reason = None
        if _words(name) & removed_words:
            reason = "excluded for safety"
        elif allergies and allergies != "none" and _words(name) & _words(allergies):
            reason = "patient allergy"
        else:
            for kind, words in BLOCKED.items():
                if any(w in low for w in words) or low.endswith(("mycin", "floxacin", "cycline")) or low.startswith("cef"):
                    reason = kind
                    break
        (rejected.append((name, reason)) if reason else ok.append({"name": name, "reason": str(item.get("reason", ""))[:80]}))
    return ok[:4], rejected