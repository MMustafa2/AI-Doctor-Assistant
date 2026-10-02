"""consultation.py - code-controlled doctor consultation (works for ANY disease).

Why the old chat repeated questions: the LLM alone decided what to ask, so it forgot what the patient
had already said. Here CODE keeps the patient's facts, remembers which questions were asked, limits the
question rounds and decides the next step:   ask  ->  diagnose (RAG agent)  ->  prescription (agent).

Uses your existing diagnosis_agent.py, prescription_agent.py, tools.py and the FAISS knowledge base.
"""
import json
import re
from dataclasses import dataclass, field

from agents import diagnosis_agent, prescription_agent
import drugs
from agents.base import run_agent
from llm import get_llm, strip_think

MAX_ROUNDS = 3                                   # at most 3 question rounds, then diagnose
GENERIC_SLOTS = ["duration", "severity", "age", "sex"]
SLOT_TEXT = {
    "complaint": "what the main problem or symptom is",
    "duration": "how long the problem has been going on",
    "severity": "how severe it is on a scale of 0 to 10",
    "age": "the patient's age",
    "sex": "whether the patient is male or female",
    "allergies": "any medicine allergies and any medicines the patient takes now",
    "pregnant": "whether she is pregnant or breastfeeding",
}
FACT_KEYS = ["complaint", "duration", "severity", "age", "sex", "allergies",
             "chronic_diseases", "current_medicines", "pregnant"]
LANG_NAME = {
    "english": "English",
    "roman_urdu": "Roman Urdu (Urdu written in English letters, like 'Aap ko kitne din se bukhar hai?')",
    "urdu": "Urdu script (اردو)",
}
GREETING = {
    "english": "Hello! How can I help you today? Please tell me about your problem.",
    "roman_urdu": "Assalam o Alaikum! Aap ko kya takleef hai? Apni takleef bataiye.",
    "urdu": "السلام علیکم! آپ کو کیا تکلیف ہے؟ اپنی تکلیف بتائیں۔",
}
OFFER = {
    "english": "Would you like me to prepare a prescription (medicine advice)?",
    "roman_urdu": "Kya aap chahte hain ke main dawai ka prescription (nuskha) tayyar kar doon?",
    "urdu": "کیا آپ چاہتے ہیں کہ میں دوا کا نسخہ تیار کر دوں؟",
}

EXTRACT_PROMPT = """You read a chat between a patient and a doctor assistant. Return ONLY one JSON object, no other text.
Keys:
complaint (main symptom or problem, short English text, or null),
duration (English text or null), severity (English text or null),
age (number as text or null), sex ("male" or "female" or null),
associated_symptoms (list of short English symptom names the patient mentioned, may be empty),
allergies (text or null; use "none" if the patient says no allergies),
chronic_diseases (text or null), current_medicines (text or null),
pregnant (text or null),
red_flag (true only for a real emergency: chest pain with arm/jaw pain, sweating or breathlessness, severe breathing trouble, fainting or unconscious, seizure, stroke signs, heavy bleeding, severe allergic reaction, high fever with stiff neck or confusion, poisoning or overdose, severe injury; otherwise false),
red_flag_reason (short English text or null),
wants_prescription (true if the patient asks for medicine, tablet or prescription, OR says yes to the assistant's offer of a prescription; otherwise false),
no_more_info (true if the patient says they do not know, have nothing more to add, or just want advice),
is_greeting_only (true if the patient message has no health information),
off_topic (true if the message is NOT about the patient's own health problem and is NOT an answer to the doctor's question, for example greetings, thanks, general chat, or asking for hospital names, places or directions; false when it answers the doctor's question or describes symptoms),
language ("english", "roman_urdu" for Urdu or Hindi written in English letters, or "urdu" for Urdu script).
Use only what the PATIENT said. Use null when something was not mentioned. Spelling may be poor or mixed."""


@dataclass
class Consultation:
    facts: dict = field(default_factory=dict)
    symptoms: list = field(default_factory=list)
    transcript: list = field(default_factory=list)       # (role, text)
    asked_slots: set = field(default_factory=set)         # a slot is asked only ONCE
    rounds: int = 0
    specific_asked: bool = False
    diagnosis: str = None
    red_flag: str = None
    red_flag_shown: bool = False
    lang: str = "english"


# ---------------------------------------------------------------- helpers
def _call(system: str, user: str, temperature: float = 0.2) -> str:
    msg = get_llm(temperature).invoke([("system", system), ("user", user)])
    return strip_think(msg.content)


_agents = {}


def _agent(name):
    if name not in _agents:
        module = diagnosis_agent if name == "diagnosis" else prescription_agent
        _agents[name] = module.build(get_llm(0.2))
    return _agents[name]


def _facts_text(c):
    lines = [f"{k}: {v}" for k, v in c.facts.items()]
    if c.symptoms:
        lines.append("associated symptoms: " + ", ".join(c.symptoms))
    if c.red_flag:
        lines.append(f"possible emergency: {c.red_flag}")
    return "\n".join(lines) or "nothing yet"


def _chat_text(c, last=None):
    rows = c.transcript[-last:] if last else c.transcript
    return "\n".join(f"{r}: {t}" for r, t in rows)


def _case_text(c):
    return f"PATIENT FACTS:\n{_facts_text(c)}\n\nCHAT:\n{_chat_text(c)}"


def _doctor_style(c):
    return (f"You are a warm, caring family doctor in Pakistan. Write in {LANG_NAME[c.lang]} only. "
            "Use simple words. Never mix languages or scripts. No emojis.")


# ---------------------------------------------------------------- understanding the patient
def _extract(c, text, last_bot):
    user = (f"Assistant's last message: {last_bot or '(none)'}\n"
            f"Patient's new message: {text}\n"
            f"Facts known so far: {json.dumps(c.facts, ensure_ascii=False)}")
    raw = _call(EXTRACT_PROMPT, user, temperature=0)
    match = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(match.group(0)) if match else {}
    except Exception:
        return {}


def _language(text, data, previous):
    if re.search(r"[\u0600-\u06FF]", text):
        return "urdu"
    lang = data.get("language")
    return lang if lang in LANG_NAME else previous


def _merge(c, data):
    """Save new facts. Returns how many NEW symptoms were added."""
    for k in FACT_KEYS:
        v = data.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            v = str(v)
        if isinstance(v, str) and v.strip() and v.strip().lower() not in {"null", "unknown", "n/a"}:
            c.facts[k] = v.strip()
    new = 0
    for s in data.get("associated_symptoms") or []:
        s = str(s).strip().lower()
        if s and s not in c.symptoms:
            c.symptoms.append(s)
            new += 1
    return new


# ---------------------------------------------------------------- the four things the doctor can do
def _ask_slots(c, slots):
    c.rounds += 1
    c.asked_slots.update(slots)
    items = "; ".join(SLOT_TEXT[s] for s in slots)
    system = (_doctor_style(c) + " Write ONE short message that asks the patient for the items below. "
              "Maximum 2 short questions, maximum 30 words, no greeting, no explanation, no bullet points. "
              "Never ask anything that is already in the known facts.")
    return _call(system, f"Known facts:\n{_facts_text(c)}\n\nAsk about: {items}")


def _ask_specific(c):
    c.rounds += 1
    c.specific_asked = True
    asked = "\n".join(t for r, t in c.transcript if r == "assistant") or "(none)"
    system = (_doctor_style(c) + " Ask exactly ONE new short question (maximum 20 words) about the most important "
              "missing detail for diagnosing this complaint, such as a key associated symptom or a danger sign. "
              "It must NOT repeat anything in the known facts or in the questions already asked.")
    return _call(system, f"Known facts:\n{_facts_text(c)}\n\nQuestions already asked:\n{asked}")


def _diagnose(c):
    prompt = (f"{_case_text(c)}\n\nTASK: find the probable diagnosis (use the tools). Write the FINAL ANSWER "
              f"for the patient in {LANG_NAME[c.lang]}, maximum 90 words, simple words: "
              "1) the most likely condition with a one-line reason, 2) two or three tests to confirm it, "
              "3) how urgent it is. Say clearly that it is a probable diagnosis, not final. Do not ask questions.")
    c.diagnosis = run_agent(_agent("diagnosis"), "diagnosis_specialist", prompt)
    return c.diagnosis


def _prescribe(c):
    context = " ".join([c.diagnosis or "", c.facts.get("complaint", "")] + c.symptoms)
    if c.red_flag:                                       # possible emergency: no medicine list at all
        allowed, removed = [], []
    else:
        allowed, removed = drugs.safety_filter(drugs.find_candidates(context), c.facts, context)
    allowed_text = drugs.format_rows(allowed) or "NONE: no suitable medicine in the database. Write 'doctor to prescribe'."
    removed_text = "; ".join(f"{g} ({why})" for g, why in removed) or "none"
    extra = ""
    if c.red_flag:
        extra = (" A possible emergency was flagged: give only safe interim advice and a strong warning "
                 "to go to the emergency department.")
    if not c.facts.get("allergies"):
        extra += " Allergies were not told: add 'confirm allergies with a doctor'."
    prompt = (f"{_case_text(c)}\n\nProbable diagnosis already given to the patient:\n{c.diagnosis}\n\n"
              f"ALLOWED MEDICINES (the ONLY medicines you may use; they match the diagnosis and passed the safety "
              f"filter):\n{allowed_text}\n\nEXCLUDED FOR SAFETY (never use): {removed_text}\n\n"
              "TASK: write the DRAFT prescription now in your required format. Do NOT ask questions. From the ALLOWED "
              "list choose the medicines that fit the patient's symptoms (not every row, and not only paracetamol). "
              "Use the generic name, the Pakistani brand names, the dose, frequency and duration exactly as in the list. "
              "Do not add any medicine that is not in the list. "
              f"Write headings and advice in {LANG_NAME[c.lang]}; keep medicine and brand names in English. "
              f"Keep it under 200 words.{extra}")
    return run_agent(_agent("prescription"), "prescription_writer", prompt)


def _emergency(c):
    system = (_doctor_style(c) + " The patient may have a medical emergency. In maximum 45 words: say clearly to call "
              "Rescue 1122 or Edhi 115 or go to the nearest emergency department now, mention in a few words why, "
              "and tell them not to wait. No questions.")
    return _call(system, f"Reason: {c.red_flag}\nFacts:\n{_facts_text(c)}")


def _chat_reply(c, text):
    """Natural LLM reply for greetings, small talk, off-topic and follow-up questions."""
    system = (_doctor_style(c) + " Reply naturally to the patient's latest message in your own words, maximum 50 words.\n"
              "- Greeting (hello, salam): greet back warmly and ask how you can help with their health. "
              "Do NOT start with Salam or a greeting unless the patient greeted first.\n"
              "- Thanks, 'no thanks', 'nothing else' or goodbye: the patient is closing the chat. Do NOT greet again "
              "and do NOT assume anything about their health. Reply with ONE short polite closing line "
              "(for example: you're welcome, take care, come back if you need help).\n"
              "- Health question: answer accurately using the facts and diagnosis below. Never invent facts. "
              "If unsure, say so and suggest seeing a doctor.\n"
              "- Hospitals, clinics, pharmacies, addresses, directions, phone numbers or place names: you have NO "
              "location data. NEVER invent any name, address, direction or number. Say you cannot look up places and "
              "suggest Google Maps, and for emergencies Rescue 1122 or Edhi 115.\n"
              "- Unclear message (a single odd word): politely ask in one short sentence what they mean. "
              "Do not repeat your earlier answers.\n"
              "- Do not mention hospitals or prescriptions unless the patient asked. If they ask for medicine, "
              "say you can prepare a prescription.")
    user = (f"Facts:\n{_facts_text(c)}\n\nDiagnosis given: {c.diagnosis or 'not yet'}\n\n"
            f"Recent chat:\n{_chat_text(c, 6)}\n\nPatient message: {text}")
    return _call(system, user, temperature=0.5)


# ---------------------------------------------------------------- the brain: what to do next
def _decide(c, data, text, new_symptoms):
    if not c.facts.get("complaint"):
        return _chat_reply(c, text)

    if data.get("red_flag") and not c.red_flag_shown:
        c.red_flag = data.get("red_flag_reason") or "serious symptoms"
        c.red_flag_shown = True
        return _emergency(c)

    if data.get("off_topic") and not data.get("wants_prescription"):
        return _chat_reply(c, text)

    if data.get("wants_prescription"):
        needed = [s for s in ["age", "allergies"] + (["pregnant"] if c.facts.get("sex") == "female" else [])
                  if not c.facts.get(s) and s not in c.asked_slots]
        if needed:
            return _ask_slots(c, needed)
        parts = [] if c.diagnosis else [_diagnose(c)]
        parts.append(_prescribe(c))
        return "\n\n".join(parts)

    if c.diagnosis and not new_symptoms:
        return _chat_reply(c, text)

    if c.rounds < MAX_ROUNDS and not data.get("no_more_info") and not c.diagnosis:
        missing = [s for s in GENERIC_SLOTS if not c.facts.get(s) and s not in c.asked_slots]
        if missing:
            return _ask_slots(c, missing[:2])
        if not c.specific_asked:
            return _ask_specific(c)

    return _diagnose(c) + "\n\n" + OFFER[c.lang]


def handle(c: Consultation, text: str):
    """One patient message in -> (reply, language) out."""
    last_bot = next((t for r, t in reversed(c.transcript) if r == "assistant"), "")
    c.transcript.append(("patient", text))
    data = _extract(c, text, last_bot)
    c.lang = _language(text, data, c.lang)
    new_symptoms = _merge(c, data)
    reply = _decide(c, data, text, new_symptoms) or GREETING[c.lang]
    c.transcript.append(("assistant", reply))
    return reply, c.lang