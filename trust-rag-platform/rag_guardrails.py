from dataclasses import dataclass
from typing import List, Dict, Any, Tuple


@dataclass
class RetrievalMatch:
    score: float
    text: str
    source: str
    chunk_id: str


@dataclass
class Decision:
    action: str  # "ANSWER" or "ABSTAIN"
    reason: str
    signals: Dict[str, float]


def compute_signals(matches: List[RetrievalMatch]) -> Dict[str, float]:
    if not matches:
        return {"top1": 0.0, "top3_avg": 0.0, "gap_1_3": 0.0, "spread_top3": 0.0}

    top1 = matches[0].score
    top3 = matches[:3]
    top3_avg = sum(m.score for m in top3) / len(top3)

    score_3 = top3[-1].score if len(top3) >= 3 else top3[-1].score
    gap_1_3 = top1 - score_3

    spread_top3 = (top3[0].score - top3[-1].score) if len(top3) >= 2 else 0.0

    return {
        "top1": round(top1, 4),
        "top3_avg": round(top3_avg, 4),
        "gap_1_3": round(gap_1_3, 4),
        "spread_top3": round(spread_top3, 4),
    }

import re

def query_intent(user_query: str) -> str:
    q = user_query.lower().strip()
    if "sla" in q:
        return "SLA_VALUE"
    if "who approves" in q or "owner" in q or "accountable" in q:
        return "OWNERSHIP"
    return "GENERAL"

def has_sla_value(text: str) -> bool:
    """
    Heuristic: SLA values usually include numbers + time units or percentage targets.
    Examples: '4 hours', '24h', '99.9%', 'within 1 day'
    """
    t = text.lower()

    # percent targets: 99%, 99.9%
    if re.search(r"\b\d{2,3}(\.\d+)?\s*%\b", t):
        return True

    # time targets: 4 hours, 15 min, 1 day, within 2 hours
    if re.search(r"\b(within\s+)?\d+(\.\d+)?\s*(sec|secs|second|seconds|min|mins|minute|minutes|hr|hrs|hour|hours|day|days)\b", t):
        return True

    # short forms: 24h, 48hr
    if re.search(r"\b\d+\s*(h|hr|hrs)\b", t):
        return True

    return False

def looks_like_glossary_only(matches) -> bool:
    # if top match is glossary, treat as weak grounding for value-based questions
    if not matches:
        return False
    return matches[0].source.lower().startswith("glossary")

def decide_answer_or_abstain(matches: List[RetrievalMatch], user_query: str = "") -> Decision:
    if not matches:
        return Decision(
            action="ABSTAIN",
            reason="No relevant passages were retrieved from the knowledge base.",
            signals=compute_signals(matches),
        )

    signals = compute_signals(matches)

    # -----------------------------
    # NEW: Evidence sufficiency gate
    # -----------------------------
    intent = query_intent(user_query)

    if intent == "SLA_VALUE":
        any_value = any(has_sla_value(m.text) for m in matches[:5])

        if not any_value:
            return Decision(
                action="ABSTAIN",
                reason="SLA mentioned, but no explicit SLA target/value found in the retrieved documents.",
                signals={**signals, "intent": "SLA_VALUE"},
            )

        if looks_like_glossary_only(matches) and not any(
            has_sla_value(m.text) for m in matches[1:5]
        ):
            return Decision(
                action="ABSTAIN",
                reason="Top evidence is a glossary definition, not an SLA policy or target.",
                signals={**signals, "intent": "SLA_VALUE"},
            )

    # -----------------------------
    # EXISTING score-based gates
    # -----------------------------
    top1 = signals["top1"]
    top3_avg = signals["top3_avg"]
    gap_1_3 = signals["gap_1_3"]

    if top1 < 0.33:
        return Decision(
            action="ABSTAIN",
            reason="Low retrieval confidence (top match score is below threshold).",
            signals=signals,
        )

    if top3_avg < 0.30:
        return Decision(
            action="ABSTAIN",
            reason="Low overall evidence (average relevance of top passages is weak).",
            signals=signals,
        )

    if gap_1_3 < 0.04:
        return Decision(
            action="ABSTAIN",
            reason="Ambiguous evidence (multiple passages appear similarly relevant).",
            signals=signals,
        )

    return Decision(
        action="ANSWER",
        reason="Sufficient evidence retrieved from the knowledge base.",
        signals=signals,
    )

def build_citations(matches: List[RetrievalMatch], max_citations: int = 3) -> List[Dict[str, Any]]:
    cites = []
    for m in matches[:max_citations]:
        snippet = m.text.strip().replace("\n", " ")
        if len(snippet) > 280:
            snippet = snippet[:280] + "..."
        cites.append({
            "source": m.source,
            "chunk_id": m.chunk_id,
            "score": round(m.score, 4),
            "snippet": snippet,
            # full text gives the LLM richer grounding than the shortened snippet
            "evidence_text": m.text,
        })
    return cites


def suggest_clarifying_question(user_query: str) -> str:
    # Simple heuristic for demo. (Later we can use LLM to generate this.)
    return (
        "Can you clarify which document area you mean (pipelines, KPIs, incident response, or data quality), "
        "and mention any specific KPI/pipeline name if applicable?"
    )
