import os
from typing import Literal
from pydantic import BaseModel, Field, ValidationError
import google.generativeai as genai

PREFERRED_MODELS = [
    "gemini-2.5-flash",          # primary (fast, supports JSON)
    "gemini-pro-latest",         # solid text fallback
]

def _pick_model():
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    names = {m.name for m in genai.list_models()}
    for m in PREFERRED_MODELS:
        if f"models/{m}" in names or m in names:
            return m
    # last-resort fallback (should exist on most accounts)
    return "gemini-pro-latest"

class ChartSpec(BaseModel):
    type: Literal["bar", "line", "area", "pie"] = "bar"
    x: str
    y: str
    aggregate: Literal["sum", "avg", "count", "min", "max"] = "sum"

class ModelResp(BaseModel):
    sql: str
    chart: ChartSpec
    explanation: str
    params: dict = Field(default_factory=dict)

SCHEMA_SUMMARY = """
tables:
  SALES(ORDER_ID STRING, ORDER_DATE DATE, REGION STRING, PRODUCT STRING, QTY NUMBER, REVENUE NUMBER)
  REGIONS(REGION STRING, COUNTRY STRING)
  PRODUCTS(PRODUCT STRING, CATEGORY STRING, COST NUMBER)
rules:
  - ONLY SELECT/WITH
  - If timeframe missing, default to last 12 months using ORDER_DATE
  - Include LIMIT 200 for raw row queries (no group)
  - Prefer aggregated answers for charts
"""

def _prompt(user_query: str) -> str:
    return f"""
You are a Snowflake SQL analyst. Convert the user's question into a single safe SQL query.
Output STRICT JSON matching:
{{
  "sql": "SELECT ...",
  "chart": {{"type":"bar|line|area|pie","x":"col","y":"col","aggregate":"sum|avg|count|min|max"}},
  "explanation": "one paragraph",
  "params": {{}}
}}

SCHEMA:
{SCHEMA_SUMMARY}

USER QUESTION:
{user_query}
"""

def synthesize_sql(user_query: str) -> ModelResp:
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY not set.")
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    model_name = _pick_model()
    model = genai.GenerativeModel(model_name)
    resp = model.generate_content(
        _prompt(user_query),
        generation_config={"response_mime_type": "application/json"}
    )
    try:
        return ModelResp.model_validate_json(resp.text)
    except ValidationError as ve:
        # If model returns non-JSON, try to coerce by extracting a JSON block
        import json, re
        m = re.search(r"\{.*\}", resp.text, re.S)
        if m:
            return ModelResp.model_validate_json(m.group(0))
        raise ve