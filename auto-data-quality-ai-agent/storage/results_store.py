from __future__ import annotations

import os
from datetime import datetime
from typing import List, Dict, Any

from supabase import Client, create_client


def _get_client() -> Client:
    """
    Create Supabase client using preferred service role key (write access) with fallbacks.
    Prefers:
      - SUPABASE_SERVICE_ROLE_KEY (recommended)
      - NEXT_PUBLIC_SUPABASE_ANON_KEY (if only anon is available)
    URL can come from SUPABASE_URL or NEXT_PUBLIC_SUPABASE_URL.
    """
    url = os.environ.get("SUPABASE_URL") or os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
    key = os.environ.get("SUPABASE_SERVICE_ROLE_KEY") or os.environ.get("SUPABASE_KEY") or os.environ.get(
        "NEXT_PUBLIC_SUPABASE_ANON_KEY"
    )
    if not url or not key:
        raise RuntimeError("SUPABASE_URL/NEXT_PUBLIC_SUPABASE_URL and a Supabase key must be set for run history storage.")
    return create_client(url, key)


def record_run(table_ref: str, row_count: int, issues: List[Dict[str, Any]]) -> int:
    """
    Persist a DQ run and its issues to Supabase.
    """
    client = _get_client()
    run_timestamp = datetime.utcnow().isoformat()
    run_payload = {
        "table_ref": table_ref,
        "run_timestamp": run_timestamp,
        "row_count": row_count,
        "issue_count": len(issues),
    }

    run_res = client.table("dq_runs").insert(run_payload).execute()
    data = getattr(run_res, "data", None) or []
    if not data or "id" not in data[0]:
        raise RuntimeError("Failed to insert dq_run into Supabase.")
    run_id = int(data[0]["id"])

    if issues:
        issue_rows = []
        for issue in issues:
            issue_rows.append(
                {
                    "run_id": run_id,
                    "column_name": issue.get("column"),
                    "issue_type": issue.get("issue_type"),
                    "severity": issue.get("severity"),
                    "rows_affected": issue.get("rows_affected"),
                    "details": issue.get("details"),
                }
            )
        client.table("dq_issues").insert(issue_rows).execute()

    return run_id


def get_recent_runs(limit: int = 20) -> List[Dict[str, Any]]:
    try:
        client = _get_client()
        res = (
            client.table("dq_runs")
            .select("id, table_ref, run_timestamp, row_count, issue_count")
            .order("id", desc=True)
            .limit(limit)
            .execute()
        )
        return getattr(res, "data", []) or []
    except Exception:
        return []


def get_issues_for_run(run_id: int) -> List[Dict[str, Any]]:
    try:
        client = _get_client()
        res = (
            client.table("dq_issues")
            .select("id, column_name, issue_type, severity, rows_affected, details")
            .eq("run_id", run_id)
            .order("id")
            .execute()
        )
        return getattr(res, "data", []) or []
    except Exception:
        return []
