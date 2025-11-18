"""NPS calculations and analytical helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable
import re
import pandas as pd

NPS_THRESHOLDS = {
    "promoter": (9, 10),
    "passive": (7, 8),
    "detractor": (0, 6),
}


@dataclass
class NPSKPI:
    """Container for high-level KPI values used in KPI cards."""

    overall_nps: float
    promoter_pct: float
    passive_pct: float
    detractor_pct: float
    total_responses: int


def categorize_nps(score: float | int) -> str:
    """Return the NPS bucket label for a given score."""

    if pd.isna(score):
        return "unknown"
    if score >= 9:
        return "promoter"
    if score >= 7:
        return "passive"
    return "detractor"


def compute_overall_nps(data: pd.DataFrame | pd.Series) -> float:
    """Compute the overall Net Promoter Score for the provided data."""

    if isinstance(data, pd.DataFrame):
        series = data["nps_score"]
    else:
        series = data
    if series.empty:
        return 0.0
    bucket_counts = series.apply(categorize_nps).value_counts()
    total = bucket_counts.sum()
    if total == 0:
        return 0.0
    promoters = bucket_counts.get("promoter", 0)
    detractors = bucket_counts.get("detractor", 0)
    return round(((promoters - detractors) / total) * 100, 2)


def compute_kpis(df: pd.DataFrame) -> NPSKPI:
    """Compute KPI summary metrics for the KPI card deck."""

    if df.empty:
        return NPSKPI(0.0, 0.0, 0.0, 0.0, 0)

    total = len(df)
    buckets = df["nps_score"].apply(categorize_nps).value_counts(normalize=True) * 100
    return NPSKPI(
        overall_nps=round(compute_overall_nps(df), 2),
        promoter_pct=round(buckets.get("promoter", 0.0), 2),
        passive_pct=round(buckets.get("passive", 0.0), 2),
        detractor_pct=round(buckets.get("detractor", 0.0), 2),
        total_responses=total,
    )


def nps_trend(df: pd.DataFrame, freq: str = "W") -> pd.DataFrame:
    """Return a time series of NPS computed by the specified frequency."""

    if df.empty:
        return pd.DataFrame(columns=["date", "nps"])

    grouped = df.set_index("date").resample(freq)["nps_score"].apply(compute_overall_nps)
    trend = grouped.reset_index().rename(columns={"nps_score": "nps"})
    trend["nps"] = trend["nps"].round(2)
    return trend


def nps_by_dimension(df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Compute NPS broken down by the requested categorical column."""

    if df.empty or dimension not in df.columns:
        return pd.DataFrame(columns=[dimension, "nps", "responses"])

    def _nps(group: pd.Series) -> float:
        return compute_overall_nps(pd.DataFrame({"nps_score": group}))

    grouped = (
        df.groupby(dimension)["nps_score"].apply(_nps).reset_index(name="nps")
    )
    grouped["responses"] = df.groupby(dimension)["nps_score"].size().values
    grouped = grouped.sort_values("nps", ascending=False)
    grouped["nps"] = grouped["nps"].round(2)
    return grouped


def subset_for_filters(
    df: pd.DataFrame,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None,
    channels: Iterable[str] | None,
    regions: Iterable[str] | None,
) -> pd.DataFrame:
    """Convenience wrapper to apply filters before computing metrics."""

    from core.data_loader import filter_dataframe  # avoid circular import

    return filter_dataframe(df, date_range=date_range, channels=channels, regions=regions)


def comparison_nps(df_a: pd.DataFrame, df_b: pd.DataFrame) -> dict[str, float]:
    """Return overall NPS metrics for comparison tiles."""

    metrics = {
        "nps_a": round(compute_overall_nps(df_a), 2),
        "nps_b": round(compute_overall_nps(df_b), 2),
        "responses_a": len(df_a),
        "responses_b": len(df_b),
    }
    metrics["delta"] = round(metrics["nps_b"] - metrics["nps_a"], 2)
    return metrics


def comparison_table(df_a: pd.DataFrame, df_b: pd.DataFrame, dimension: str) -> pd.DataFrame:
    """Return a side-by-side breakdown used in the comparison tab."""

    breakdown_a = nps_by_dimension(df_a, dimension).set_index(dimension)
    breakdown_b = nps_by_dimension(df_b, dimension).set_index(dimension)
    combined = breakdown_a.join(
        breakdown_b,
        lsuffix="_a",
        rsuffix="_b",
        how="outer",
    ).fillna(0)
    combined["delta_nps"] = combined["nps_b"] - combined["nps_a"]
    combined["delta_responses"] = combined["responses_b"] - combined["responses_a"]
    combined.reset_index(inplace=True)
    for column in ["nps_a", "nps_b", "delta_nps"]:
        if column in combined:
            combined[column] = combined[column].astype(float).round(2)
    return combined


STOPWORDS = {
    "the",
    "and",
    "a",
    "to",
    "of",
    "in",
    "is",
    "for",
    "on",
    "it",
    "this",
    "that",
    "was",
    "with",
    "but",
    "are",
    "not",
    "be",
    "have",
}


def top_keywords(df: pd.DataFrame, top_n: int = 12) -> list[dict[str, int]]:
    """Return the most frequent keywords from survey comments."""

    if df.empty or "comment_text" not in df:
        return []
    tokens: dict[str, int] = {}
    for comment in df["comment_text"].dropna():
        words = re.findall(r"[a-zA-Z']+", str(comment).lower())
        for word in words:
            if len(word) <= 2 or word in STOPWORDS:
                continue
            tokens[word] = tokens.get(word, 0) + 1
    sorted_tokens = sorted(tokens.items(), key=lambda item: item[1], reverse=True)[:top_n]
    return [{"word": word, "count": count} for word, count in sorted_tokens]
