"""Multimodal insight synthesis for videos, documents, logs, and images."""

from __future__ import annotations

from typing import Dict, List


def _merge_blobs(text_blobs: List[str], transcript_snippets: str) -> str:
    content = []
    if transcript_snippets:
        content.append(f"Transcript highlights:\n{transcript_snippets}")
    for idx, blob in enumerate(text_blobs, start=1):
        label = f"Source {idx}"
        content.append(f"{label}:\n{blob}")
    return "\n\n".join(content)


def _format_transcript_snippets(transcript_df, limit: int = 12) -> str:
    if transcript_df is None or transcript_df.empty:
        return ""
    snippets = []
    for row in transcript_df.head(limit).itertuples():
        snippets.append(f"[{row.start:.1f}s-{row.end:.1f}s] {row.text}")
    return "\n".join(snippets)


def generate_multimodal_bundle(
    text_blobs: List[str],
    transcript_df,
    llm_client,
    max_words: int = 260,
) -> Dict:
    """Return insights, action items, and a narration script spanning all uploads."""
    transcript_snippets = _format_transcript_snippets(transcript_df)
    if not llm_client or not getattr(llm_client, "available", False):
        return {
            "insights": ["AI unavailable. Add your OpenAI API key to generate insights."],
            "actions": [],
            "risks": [],
            "narration_script": "",
            "email_summary": "",
        }

    context = _merge_blobs(text_blobs, transcript_snippets)
    system_prompt = (
        "You are an enterprise analyst who blends meeting transcripts, CSV summaries, PDFs, logs, and images "
        "into a concise decision brief. Keep language executive-friendly and outcome-focused."
    )
    user_prompt = (
        f"Context from uploads:\n{context}\n\n"
        "Return JSON with fields:\n"
        "- insights: array of 4-7 bullet strings.\n"
        "- actions: array of objects with item, owner, due fields.\n"
        "- risks: array of short bullets (optional).\n"
        f"- narration_script: <= {max_words} words summarizing insights for a voice avatar.\n"
        "- email_summary: <=160 words, actionable tone, to send to stakeholders.\n"
    )
    response = llm_client.response_json(system_prompt, user_prompt, max_output_tokens=900)
    insights = response.get("insights") or []
    if isinstance(insights, str):
        insights = [insights]
    actions = response.get("actions") or response.get("action_items") or []
    risks = response.get("risks") or []
    if isinstance(risks, str):
        risks = [risks]
    return {
        "insights": insights,
        "actions": actions,
        "risks": risks,
        "narration_script": response.get("narration_script", ""),
        "email_summary": response.get("email_summary", ""),
    }
