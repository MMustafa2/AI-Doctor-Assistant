# Pakistan AI Doctor Assistant (RAG + Qwen on Groq + FAISS + create_react_agent)

## Setup
1. Put your existing `medical_apis.py` in this folder (next to `config.py`).
2. `pip install -r requirements.txt`
3. Copy `.env.example` to `.env`, add `GROQ_API_KEY` (and ICD keys if you have them).
4. (Optional) Put NIH / WHO guideline PDFs in `knowledge_base/`.
5. `python build_index.py`   (load APIs -> chunk -> embed -> FAISS; run once)
6. `python main.py`          (chat with the doctor)

## Files
| File | Job |
|---|---|
| config.py | all settings: model, chunk size, disease/drug lists |
| medical_apis.py | one function per API (yours) |
| rag/loaders.py | 1. load API + PDF data |
| rag/chunking.py | 2. chunking |
| rag/embeddings.py | 3. embeddings |
| rag/vector_store.py | 4. FAISS save/load |
| rag/retriever.py | 5. retrieval |
| llm.py | Qwen via Groq |
| tools.py | tools for agents |
| agents/base.py | AgentSpec (role/work/rules) + builder |
| agents/symptom_agent.py | triage / history |
| agents/diagnosis_agent.py | disease identification |
| agents/prescription_agent.py | draft prescription |
| agents/doctor_agent.py | main doctor, talks to patient, calls the 3 agents |
| build_index.py / main.py | run these |
