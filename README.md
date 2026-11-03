# Pakistan AI Doctor Assistant
RAG + Qwen on Groq + FAISS + multi-agent (symptom / diagnosis / prescription / doctor)

> ## ⚠️ Important - please read first
> **This is an AI draft assistant. It is NOT a licensed doctor.** It can make mistakes.
> Its answers are general health information, not a medical diagnosis or a prescription.
>
> - **Never use it in an emergency.** Go to the nearest hospital emergency department or call an
>   ambulance (Rescue 1122 where available, or Edhi 115).
> - **`medicines_pk.csv`: all brands and doses must be verified by a licensed doctor or pharmacist**
>   before anyone relies on them.
> - Any "prescription" it produces is only a **draft** and must be reviewed and approved by a
>   licensed doctor.

## What it does
A patient describes their problem in English, Urdu or Roman Urdu. The **doctor agent** talks to the
patient and calls three helper agents, which use retrieved guideline text and medical APIs:

```
Patient
  -> Doctor agent  -> Symptom agent        (history, urgency / triage)
                   -> Diagnosis agent      (possible conditions)
                   -> Prescription agent   (draft only, needs doctor approval)
  -> Answer with sources
```

## Setup

1. **Get the code** and open a terminal in the project folder.
2. **Create a virtual environment** (Python 3.10 or newer):
   ```bash
   python -m venv .venv
   # Windows PowerShell:  .venv\Scripts\Activate.ps1
   # macOS / Linux:       source .venv/bin/activate
   ```
3. **Install the packages:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Create your `.env` file.** Copy `.env.example` to `.env` and add your `GROQ_API_KEY`
   (a free key from https://console.groq.com):
   ```bash
   # Windows:        copy .env.example .env
   # macOS / Linux:  cp .env.example .env
   ```
5. **(Optional) Add your own guideline documents** to `knowledge_base/`
   (read the licence note below first).
6. **Build the vector database** (run once, and again whenever the documents change):
   ```bash
   python build_index.py
   ```
7. **Start the app:**
   ```bash
   streamlit run app.py
   ```
   For a terminal chat instead of the web page, run `python main.py`.

The first run downloads the embedding model, so it needs internet and may take a few minutes.

## Environment variables (`.env`)
`.env.example` should contain:
```
GROQ_API_KEY=your_groq_key_here

# Optional (only if you use these APIs)
ICD_CLIENT_ID=
ICD_CLIENT_SECRET=
OPENFDA_API_KEY=
OPENALEX_API_KEY=
```
- `GROQ_API_KEY` is required.
- ICD-11 keys are free: register at https://icd.who.int/icdapi
- **Never commit your real `.env` file** and never share your keys.

## Knowledge base and licences
- **Do not upload other people's PDFs (NIH, WHO, DRAP, etc.) to a public repository** unless their
  licence allows redistribution. Many documents are "all rights reserved".
- Keep them only on your own computer. Each user should download the documents themselves.
- Recommended `.gitignore` lines (adjust the folder names to your project):
  ```
  .env
  knowledge_base/*.pdf
  faiss_index/
  ```
- Check the date of every guideline. Old versions get replaced (for example, immunization schedules).
- Only send medical terms to the external APIs, never a patient's name, CNIC or phone number.

## Medicine data (`medicines_pk.csv`)
- Brand names, active ingredients, strengths and doses in this file **must be checked by a licensed
  doctor or pharmacist** before use.
- Do not copy the DRAP registered-drug list in bulk without permission. DRAP states that its
  online list is provisional and not for research or citation.

## Files
| File | Job |
|---|---|
| `app.py` | Streamlit web interface |
| `main.py` | Terminal chat |
| `build_index.py` | Load APIs and PDFs -> chunk -> embed -> FAISS (run once) |
| `config.py` | All settings: model, chunk size, disease and drug lists |
| `medical_apis.py` | One function per medical API |
| `medicines_pk.csv` | Medicine and brand list (**needs expert verification**) |
| `rag/loaders.py` | 1. Load API and PDF data |
| `rag/chunking.py` | 2. Chunking |
| `rag/embeddings.py` | 3. Embeddings |
| `rag/vector_store.py` | 4. FAISS save and load |
| `rag/retriever.py` | 5. Retrieval |
| `llm.py` | Qwen through Groq |
| `tools.py` | Tools the agents can call |
| `agents/base.py` | `AgentSpec` (role, work, rules) and the agent builder |
| `agents/symptom_agent.py` | Triage and history |
| `agents/diagnosis_agent.py` | Possible conditions |
| `agents/prescription_agent.py` | Draft prescription |
| `agents/doctor_agent.py` | Main doctor: talks to the patient and calls the three agents |
| `.env.example` | Template for your `.env` |
| `requirements.txt` | Python packages |

## Troubleshooting
- **`ModuleNotFoundError`**: the package is not installed in the Python you are running. Activate
  the `.venv` and run `pip install -r requirements.txt` again.
- **`GROQ_API_KEY` error**: check that `.env` exists (not just `.env.example`) and contains the key.
- **Index not found**: run `python build_index.py` first.
- **Groq model not found**: Groq changes its model list. Check the current Qwen model ID in your
  Groq console and update it in `config.py`.

## Limitations
- Answers depend on the documents you provide and on retrieval quality. Wrong or outdated documents
  give wrong answers.
- Urdu and Roman Urdu quality must be checked by Urdu-speaking doctors.
- It has not been clinically validated.