from __future__ import annotations

import json
import os
from typing import Dict, List, Any

from openai import OpenAI, OpenAIError

from .prompt_builder import build_prompt


def _fallback_response(message: str) -> Dict[str, Any]:
    return {
        "overall_summary": message,
        "issue_explanations": [],
        "recommended_sql_fixes": "No AI suggestions available.",
        "recommended_python_fixes": "No AI suggestions available.",
        "recommended_rules": [],
    }


def generate_ai_analysis(profiling_dict: Dict, issues: List[Dict], user_notes: str = "") -> Dict[str, Any]:
    """
    Call OpenAI once to generate structured analysis based on metadata only.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return _fallback_response("OPENAI_API_KEY not configured; skipping AI analysis.")

    client = OpenAI(api_key=api_key)
    prompt = build_prompt(profiling_dict, issues, user_notes=user_notes or None)
    system_instructions = (
        "You are a senior data engineer specializing in data quality. "
        "You only have access to profiling metadata and issue summaries. "
        "Never ask for raw data. Return a JSON object with keys: "
        "overall_summary, issue_explanations (list of {issue_id, explanation}), "
        "recommended_sql_fixes (string, may include markdown), "
        "recommended_python_fixes (string, may include markdown), "
        "recommended_rules (list of strings)."
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_instructions},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        content = response.choices[0].message.content if response.choices else ""
        if not content:
            return _fallback_response("Empty response from OpenAI.")

        try:
            # Handle content wrapped in markdown fences.
            cleaned = content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                # Remove potential language hint e.g., ```json
                if "\n" in cleaned:
                    cleaned = cleaned.split("\n", 1)[1]
            parsed = json.loads(cleaned)
            return {
                "overall_summary": parsed.get("overall_summary", ""),
                "issue_explanations": parsed.get("issue_explanations", []),
                "recommended_sql_fixes": parsed.get("recommended_sql_fixes", ""),
                "recommended_python_fixes": parsed.get("recommended_python_fixes", ""),
                "recommended_rules": parsed.get("recommended_rules", []),
            }
        except json.JSONDecodeError:
            # If model did not follow JSON, return plain text in summary.
            return {
                "overall_summary": content,
                "issue_explanations": [],
                "recommended_sql_fixes": "",
                "recommended_python_fixes": "",
                "recommended_rules": [],
            }
    except OpenAIError as exc:
        return _fallback_response(f"OpenAI error: {exc}")
