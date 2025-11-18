"""Data loading and validation utilities for survey files."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import logging

import pandas as pd

REQUIRED_COLUMNS: tuple[str, ...] = (
    "response_id",
    "date",
    "channel",
    "region",
    "nps_score",
    "comment_text",
    "segment",
    "product_category",
)


class DataValidationError(Exception):
    """Raised when uploaded data is missing required information."""


@dataclass
class DataHealth:
    """Summarizes data completeness metrics."""

    missing_nps_pct: float
    empty_comment_pct: float


def load_survey_data(path: str | Path) -> pd.DataFrame:
    """Load a CSV file from disk and return a cleaned DataFrame."""

    logging.info("Loading survey data from %s", path)
    df = pd.read_csv(path)
    return prepare_survey_dataframe(df)


def prepare_survey_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and coerce raw survey data into a standard shape."""

    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_columns:
        raise DataValidationError(
            f"Missing required columns: {', '.join(missing_columns)}"
        )

    normalized = df.copy()
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized["nps_score"] = (
        pd.to_numeric(normalized["nps_score"], errors="coerce")
        .round()
        .clip(lower=0, upper=10)
    )
    normalized["comment_text"] = (
        normalized["comment_text"].fillna(" ").astype(str).str.strip()
    )

    optional_columns = {
        "country": "Unknown",
        "store_id": "N/A",
    }
    for column, default in optional_columns.items():
        if column not in normalized.columns:
            normalized[column] = default

    normalized.dropna(subset=["date", "nps_score"], inplace=True)
    normalized["nps_score"] = normalized["nps_score"].astype(int)
    normalized.sort_values("date", inplace=True)
    normalized.reset_index(drop=True, inplace=True)
    return normalized


def dataframe_profile(df: pd.DataFrame) -> dict[str, object]:
    """Return simple profiling stats for the UI info panel."""

    if df.empty:
        return {
            "rows": 0,
            "date_range": "N/A",
            "channels": 0,
            "regions": 0,
        }

    date_min = df["date"].min().date()
    date_max = df["date"].max().date()
    formatted_range = f"{date_min:%b %d, %Y} → {date_max:%b %d, %Y}"
    return {
        "rows": len(df),
        "date_range": formatted_range,
        "channels": df["channel"].nunique(),
        "regions": df["region"].nunique(),
    }


def data_health(df: pd.DataFrame) -> DataHealth:
    """Calculate missingness indicators for NPS and comments."""

    if df.empty:
        return DataHealth(0.0, 0.0)

    missing_nps = df["nps_score"].isna().mean() * 100
    empty_comments = df["comment_text"].replace("", pd.NA).isna().mean() * 100
    return DataHealth(missing_nps_pct=missing_nps, empty_comment_pct=empty_comments)


def filter_dataframe(
    df: pd.DataFrame,
    date_range: tuple[pd.Timestamp, pd.Timestamp] | None = None,
    channels: Iterable[str] | None = None,
    regions: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Apply common filters used throughout the UI."""

    if df.empty:
        return df

    filtered = df
    if date_range:
        start, end = date_range
        filtered = filtered[(filtered["date"] >= start) & (filtered["date"] <= end)]
    if channels:
        filtered = filtered[filtered["channel"].isin(list(channels))]
    if regions:
        filtered = filtered[filtered["region"].isin(list(regions))]
    return filtered.copy()
