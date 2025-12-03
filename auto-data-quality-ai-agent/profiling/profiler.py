from __future__ import annotations

import pandas as pd
from typing import Any, Dict

from supabase import Client

from supabase_client.metadata import get_row_count


def _infer_type(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "numeric"
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
        return "date"
    if pd.api.types.is_bool_dtype(series):
        return "bool"
    return "text"


def profile_table(
    client: Client,
    table: str,
    schema: str = "public",
    sample_size: int = 5000,
    distinct_cap: int = 10000,
) -> Dict[str, Any]:
    """
    Profile a Supabase table using a limited sample to compute metadata-only statistics.
    Raw row values never leave the application boundary.
    """
    table_ref = f"{schema}.{table}" if schema else table
    row_count = get_row_count(client, table)

    res = client.table(table).select("*").limit(sample_size).execute()
    data = getattr(res, "data", []) or []
    sample_df = pd.DataFrame(data)

    profiling: Dict[str, Any] = {
        "table_ref": table_ref,
        "row_count": row_count,
        "columns": {},
        "approx_duplicate_rows": 0,
    }

    if sample_df.empty:
        return profiling

    duplicate_count = int(sample_df.duplicated().sum())
    profiling["approx_duplicate_rows"] = duplicate_count

    for col in sample_df.columns:
        series = sample_df[col]
        col_type = _infer_type(series)
        col_profile: Dict[str, Any] = {
            "type": col_type,
            "null_pct": float(series.isna().mean()),
            "distinct_count": int(series.nunique(dropna=True)),
        }

        # Apply cap to avoid heavy counting downstream.
        if col_profile["distinct_count"] > distinct_cap:
            col_profile["distinct_count"] = distinct_cap
            col_profile["distinct_count_capped"] = True

        if col_type == "numeric":
            numeric_series = pd.to_numeric(series, errors="coerce")
            col_profile.update(
                {
                    "min": float(numeric_series.min(skipna=True))
                    if not numeric_series.dropna().empty
                    else None,
                    "max": float(numeric_series.max(skipna=True))
                    if not numeric_series.dropna().empty
                    else None,
                    "negative_count": int((numeric_series < 0).sum(skipna=True)),
                    "zero_count": int((numeric_series == 0).sum(skipna=True)),
                }
            )
        elif col_type == "date":
            datetime_series = pd.to_datetime(series, errors="coerce")
            col_profile.update(
                {
                    "min": datetime_series.min(skipna=True).isoformat()
                    if not datetime_series.dropna().empty
                    else None,
                    "max": datetime_series.max(skipna=True).isoformat()
                    if not datetime_series.dropna().empty
                    else None,
                }
            )

        profiling["columns"][col] = col_profile

    return profiling
