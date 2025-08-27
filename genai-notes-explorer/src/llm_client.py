# src/llm_client.py
import os
from typing import List
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

def get_client() -> OpenAI:
    return OpenAI()

def chat_complete(system: str, user: str, model: str | None = None) -> str:
    client = get_client()
    model = model or os.getenv("CHAT_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=float(os.getenv("TEMP", "0.2")),
        max_tokens=int(os.getenv("MAX_TOKENS", "600")),
    )
    return resp.choices[0].message.content.strip()