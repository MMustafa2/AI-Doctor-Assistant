"""config.py - all settings in ONE place. Change topics, model names, chunk sizes here."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"          # put NIH / WHO guideline PDFs here
VECTOR_DIR = BASE_DIR / "vector_store" / "faiss_index"    # FAISS index is saved here

# ---- LLM (Groq) ----
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")    # check console.groq.com/docs/models for current Qwen id
LLM_TEMPERATURE = 0.3

# ---- Embeddings (multilingual, so Urdu / Roman Urdu / English all work) ----
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ---- Chunking / retrieval ----
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 4

DEBUG = os.getenv("DEBUG", "0") == "1"                    # prints the agents' ReAct steps

# ---- What build_index.py downloads from the APIs (medical words only, no patient data) ----
DISEASE_TOPICS = [
    "dengue", "malaria", "typhoid", "hepatitis A", "hepatitis B", "hepatitis C", "tuberculosis",
    "diabetes", "hypertension", "asthma", "pneumonia", "diarrhea", "gastroenteritis",
    "urinary tract infection", "anemia", "heat stroke", "influenza", "COVID-19",
    "migraine", "gastritis", "typhoid fever", "cholera", "scabies", "chickenpox",
]
DRUG_TOPICS = [
    "paracetamol", "ibuprofen", "aspirin", "metformin", "amlodipine", "omeprazole",
    "cetirizine", "salbutamol", "metronidazole", "amoxicillin", "azithromycin", "ors",
]
WHO_INDICATORS = {"WHOSIS_000001": "Life expectancy at birth"}   # code -> name
