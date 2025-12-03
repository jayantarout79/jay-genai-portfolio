from __future__ import annotations

from typing import List, Tuple

from supabase import Client


def list_tables(client: Client, schema: str = "public") -> List[str]:
    """
    List tables in a schema using information_schema. Requires service role or adequate privileges.
    """
    try:
        res = (
            client.table("information_schema.tables")
            .select("table_name")
            .eq("table_schema", schema)
            .execute()
        )
        data = getattr(res, "data", []) or []
        return [row["table_name"] for row in data]
    except Exception:
        return []


def get_row_count(client: Client, table: str) -> int:
    res = client.table(table).select("*", count="exact").limit(0).execute()
    return int(getattr(res, "count", 0) or 0)


def get_columns(client: Client, table: str, schema: str = "public") -> List[Tuple[str, str]]:
    """
    Fetch column names and types from information_schema if accessible.
    """
    try:
        res = (
            client.table("information_schema.columns")
            .select("column_name,data_type")
            .eq("table_schema", schema)
            .eq("table_name", table)
            .order("ordinal_position")
            .execute()
        )
        data = getattr(res, "data", []) or []
        return [(row["column_name"], row["data_type"]) for row in data]
    except Exception:
        return []
