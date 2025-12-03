from __future__ import annotations

from typing import Dict, List, Optional


def build_prompt(profiling_dict: Dict, issues: List[Dict], user_notes: Optional[str] = None) -> str:
    """
    Build a concise prompt using only profiling metadata and issue summaries.
    """
    table_ref = profiling_dict.get("table_ref", "Unknown")
    row_count = profiling_dict.get("row_count", 0)
    columns = profiling_dict.get("columns", {})

    lines = [
        "You are a senior data engineer focused on data quality.",
        "You only see metadata and issue summaries, not raw data.",
        "Do not request raw data. Work strictly with the provided information.",
        "",
        f"Table: {table_ref}",
        f"Row count: {row_count}",
        f"Column count: {len(columns)}",
        "",
        "Column summaries:",
    ]

    for col, stats in columns.items():
        lines.append(
            f"- {col}: type={stats.get('type')}, null_pct={stats.get('null_pct', 0):.2%}, "
            f"distinct={stats.get('distinct_count')}"
        )

    lines.append("")
    if user_notes:
        lines.append("User-provided context / constraints (treat as authoritative):")
        lines.append(user_notes.strip())
        lines.append("")
    lines.append("Detected data-quality issues:")
    if not issues:
        lines.append("- None detected by heuristics; suggest potential checks.")
    else:
        for issue in issues:
            lines.append(
                f"- {issue['issue_id']}: {issue['issue_type']} on {issue['column']} "
                f"(severity={issue['severity']}, rows={issue['rows_affected']}): {issue['details']}"
            )

    lines.extend(
        [
            "",
            "Tasks:",
            "1) Explain likely root causes for the issues.",
            "2) Group related issues.",
            "3) Assess severity (HIGH/MEDIUM/LOW) with justification.",
            "4) Suggest SQL fixes.",
            "5) Suggest Python/pandas fixes for pipelines.",
            "6) Propose additional validation rules to prevent recurrences.",
        ]
    )
    return "\n".join(lines)
