"""llm.py - GPT-OSS through Groq (free tier)."""
import re

from langchain_groq import ChatGroq

import config

MODEL_NAME = "openai/gpt-oss-20b"      # or "openai/gpt-oss-120b"


def get_llm(temperature: float = config.LLM_TEMPERATURE) -> ChatGroq:
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY missing. Put it in .env")
    return ChatGroq(model=MODEL_NAME, temperature=temperature,
                    api_key=config.GROQ_API_KEY,
                    max_tokens=1500,
                    reasoning_effort="low",     # less thinking = faster, fewer tokens
                    timeout=60, max_retries=3)


def strip_think(text: str) -> str:
    """Remove <think>...</think> if the model prints its thinking."""
    return re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()