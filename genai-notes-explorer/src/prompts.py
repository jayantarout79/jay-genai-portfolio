# src/prompts.py
SYSTEM_PROMPT = """You are a precise assistant that answers ONLY using the provided context.
Rules:
- Start with 3–5 bullet points of the direct answer.
- Then add a short 'Why this matters' note (1–2 lines).
- Cite sources inline as [source:filename#chunk].
- If unsure, say what’s missing; never invent facts."""

USER_PROMPT = """Question:
{question}

Context:
{context}

Instructions:
- Use only the context above.
- Include 2–4 inline citations in your bullets where relevant."""