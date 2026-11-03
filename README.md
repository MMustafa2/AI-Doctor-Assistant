# Pakistan AI Doctor Assistant

An AI chatbot that talks like a doctor in **English, Urdu or Roman Urdu**. It asks short questions, finds a
**probable** disease from medical sources (RAG), and writes a **draft** prescription from a medicine list.

> ⚠️ **AI draft only. NOT a licensed doctor.** It can make mistakes. In an emergency go to a hospital or call
> Rescue 1122 / Edhi 115. Any prescription must be checked by a PMDC-registered doctor.

## How it works
1. The patient describes the problem.
2. The assistant asks a few questions (each question only once).
3. It finds the probable disease using a knowledge base (FAISS + embeddings).
4. On request it writes a draft prescription using only medicines from `data/medicines_pk.csv`, after a safety
   filter (allergy, age, pregnancy, dengue, kidney / liver problems).
5. Serious cases: it tells the patient to go to the hospital and call Rescue 1122.

## Setup
```bash
pip install -r requirements.txt
```
1. Copy `.env.example` to `.env` and add your free key from https://console.groq.com:
   ```
   GROQ_API_KEY=your_key_here
   ```
2. Build the knowledge base (run once, needs internet):
   ```bash
   python build_index.py
   ```
3. Start the app:
   ```bash
   streamlit run app.py
   ```

## Main files
| File | What it does |
|---|---|
| `app.py` | Web chat (Streamlit) |
| `Consultation.py` | The conversation: ask, diagnose, prescribe, safety rules |
| `drugs.py` + `data/medicines_pk.csv` | Medicine list and safety filter |
| `agents/` | Diagnosis and prescription agents |
| `rag/`, `build_index.py` | Knowledge base (load, chunk, embed, FAISS) |
| `llm.py` | The LLM on Groq (change the model name here) |
| `knowledge_base/` | Your guideline files and `treatment_guide.md` |

## Important notes
- **Verify the data:** brands, doses and treatments in `medicines_pk.csv` and `treatment_guide.md` must be
  checked by a doctor or pharmacist. The medicine list is a small starter list.
- **Never upload your `.env` file** or other people's copyrighted PDFs to GitHub.
- Patient messages are sent to Groq. The chat is not saved to disk.
- Not clinically validated.

## Common problems
- **Index not found:** run `python build_index.py`.
- **Groq "model not found":** change `MODEL_NAME` in `llm.py` to a model your key can use.
- **Rate limit error (429):** wait a minute and try again (free tier).