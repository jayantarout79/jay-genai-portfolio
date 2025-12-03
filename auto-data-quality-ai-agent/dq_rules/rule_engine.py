from __future__ import annotations

from datetime import datetime
from typing import Dict, List, Optional
from uuid import uuid4


def _issue(
    column: str,
    issue_type: str,
    severity: str,
    rows_affected: int,
    details: str,
) -> Dict:
    return {
        "issue_id": str(uuid4()),
        "column": column,
        "issue_type": issue_type,
        "severity": severity,
        "rows_affected": rows_affected,
        "details": details,
    }


def run_dq_checks(profiling_dict: Dict, ignore_null_columns: Optional[List[str]] = None) -> List[Dict]:
    """
    Run heuristic data-quality checks based on profiling metadata only.
    """
    issues: List[Dict] = []
    if not profiling_dict:
        return issues

    row_count = profiling_dict.get("row_count", 0) or 0
    columns = profiling_dict.get("columns", {})
    ignore_null_set = {c.lower() for c in (ignore_null_columns or [])}

    duplicate_rows = profiling_dict.get("approx_duplicate_rows", 0) or 0
    if duplicate_rows > 0:
        issues.append(
            _issue(
                column="__table__",
                issue_type="DUPLICATE_ROWS",
                severity="MEDIUM" if duplicate_rows / max(row_count, 1) < 0.05 else "HIGH",
                rows_affected=int(duplicate_rows),
                details=f"Approx. {duplicate_rows} duplicate rows detected based on sampled data.",
            )
        )

    today = datetime.utcnow().date()
    for col_name, stats in columns.items():
        col_type = stats.get("type")
        null_pct = float(stats.get("null_pct", 0.0))
        distinct_count = int(stats.get("distinct_count", 0))

        if null_pct > 0.2 and col_name.lower() not in ignore_null_set:
            severity = "HIGH" if null_pct > 0.5 else "MEDIUM"
            rows_affected = int(null_pct * row_count)
            issues.append(
                _issue(
                    column=col_name,
                    issue_type="HIGH_NULL_PERCENTAGE",
                    severity=severity,
                    rows_affected=rows_affected,
                    details=f"{null_pct:.0%} of rows are null in {col_name}.",
                )
            )

        lowered = col_name.lower()
        amount_like = any(k in lowered for k in ["amount", "price", "qty", "quantity", "cost", "revenue", "gross", "net"])
        flag_like = any(k in lowered for k in ["flag", "is_", "has_", "active"])
        status_like = any(k in lowered for k in ["status", "state", "category", "type"])

        if col_type == "numeric":
            negative_count = int(stats.get("negative_count", 0) or 0)
            zero_count = int(stats.get("zero_count", 0) or 0)
            if amount_like and negative_count > 0:
                issues.append(
                    _issue(
                        column=col_name,
                        issue_type="NEGATIVE_VALUES",
                        severity="HIGH",
                        rows_affected=negative_count,
                        details=f"{negative_count} rows with {col_name} < 0, but column name suggests non-negative values.",
                    )
                )
            if amount_like and zero_count > 0:
                issues.append(
                    _issue(
                        column=col_name,
                        issue_type="SUSPICIOUS_ZERO_VALUES",
                        severity="MEDIUM",
                        rows_affected=zero_count,
                        details=f"{zero_count} rows with {col_name} = 0 in an amount-like column.",
                    )
                )

        if flag_like and distinct_count > 10:
            issues.append(
                _issue(
                    column=col_name,
                    issue_type="FLAG_COLUMN_WITH_HIGH_CARDINALITY",
                    severity="MEDIUM",
                    rows_affected=row_count,
                    details=f"{col_name} looks like a flag but has {distinct_count} distinct values.",
                )
            )

        if status_like and distinct_count > 50:
            issues.append(
                _issue(
                    column=col_name,
                    issue_type="STATUS_WITH_TOO_MANY_VALUES",
                    severity="LOW",
                    rows_affected=row_count,
                    details=f"{col_name} has {distinct_count} distinct values; consider enforcing an enum.",
                )
            )

        if col_type == "date":
            max_value = stats.get("max")
            if max_value:
                try:
                    max_date = datetime.fromisoformat(str(max_value)).date()
                    if max_date > today:
                        issues.append(
                            _issue(
                                column=col_name,
                                issue_type="FUTURE_DATES",
                                severity="HIGH",
                                rows_affected=row_count,
                                details=f"{col_name} contains dates in the future (max={max_date.isoformat()}).",
                            )
                        )
                except ValueError:
                    # Ignore parsing issues; profiling already best-effort.
                    continue

    return issues
