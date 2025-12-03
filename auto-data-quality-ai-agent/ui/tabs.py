from __future__ import annotations

import streamlit as st
import pandas as pd
import json
from typing import Optional, List

from profiling.profiler import profile_table
from dq_rules.rule_engine import run_dq_checks
from ai_agent.reasoning import generate_ai_analysis
from storage.results_store import record_run, get_recent_runs, get_issues_for_run
from ai_agent.emailer import send_run_email


def _store_session(profiling, issues, ai_analysis, run_id):
    st.session_state["profiling"] = profiling
    st.session_state["issues"] = issues
    st.session_state["ai_analysis"] = ai_analysis
    st.session_state["latest_run_id"] = run_id


def _parse_ignore_columns(raw: str) -> List[str]:
    return [c.strip() for c in raw.split(",") if c.strip()]


def _clean_sql(sql_text: str) -> str:
    """
    Strip markdown fences and return raw SQL for execution.
    """
    cleaned = sql_text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        lines = [ln for ln in lines if not ln.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _stringify_fixes(fixes) -> str:
    """Normalize AI fixes that may be str/list/dict for safe rendering/searching."""
    if fixes is None:
        return ""
    if isinstance(fixes, str):
        return fixes
    if isinstance(fixes, list):
        parts = []
        for item in fixes:
            if isinstance(item, dict):
                parts.append(json.dumps(item, indent=2))
            else:
                parts.append(str(item))
        return "\n\n".join(parts)
    if isinstance(fixes, dict):
        return json.dumps(fixes, indent=2)
    return str(fixes)


def render_connect_and_run_tab(client, selected_schema: Optional[str], selected_table: Optional[str]):
    st.subheader("Connect & Run Checks")
    if not client:
        st.warning("Supabase connection unavailable. Check environment variables.")
        return
    if not (selected_schema and selected_table):
        st.info("Provide a schema and table in the sidebar to run checks.")
        return

    col1, col2 = st.columns(2)
    with col1:
        user_notes = st.text_area(
            "Optional: Provide context/instructions for AI (e.g., columns where nulls are expected)",
            value=st.session_state.get("ai_notes", ""),
            placeholder="Example: discount_amount can be null; status can have many enums.",
        )
    with col2:
        ignore_null_raw = st.text_input(
            "Columns to ignore for high-null checks (comma separated)",
            value=st.session_state.get("ignore_null_columns_raw", ""),
            placeholder="discount_amount, optional_field",
        )

    if st.button("Run Data Quality Analysis", type="primary"):
        st.session_state["ai_notes"] = user_notes
        st.session_state["ignore_null_columns_raw"] = ignore_null_raw
        ignore_null_columns = _parse_ignore_columns(ignore_null_raw)
        email_status = (False, "Email not attempted")
        with st.spinner("Profiling table and running checks..."):
            profiling = profile_table(client, selected_table, selected_schema)
            issues = run_dq_checks(profiling, ignore_null_columns=ignore_null_columns)
            ai_analysis = generate_ai_analysis(profiling, issues, user_notes=user_notes)
            run_id = record_run(profiling.get("table_ref", ""), profiling.get("row_count", 0), issues)
            _store_session(profiling, issues, ai_analysis, run_id)
            # Fire-and-forget email; errors are surfaced as info/warning.
            email_status = send_run_email(profiling, issues, ai_analysis)
        success_msg = f"Analysis complete for {profiling.get('table_ref')} with {len(issues)} issue(s). Run ID: {run_id}"
        if email_status[0]:
            st.success(success_msg + " | Email notification sent.")
        else:
            st.success(success_msg)
            st.info(f"Email not sent: {email_status[1]}")


def render_profile_and_issues_tab():
    st.subheader("Profile & Issues")
    profiling = st.session_state.get("profiling")
    issues = st.session_state.get("issues", [])
    if not profiling:
        st.info("No profiling results yet. Run the analysis first.")
        return

    st.metric("Row count", profiling.get("row_count", 0))
    st.metric("Column count", len(profiling.get("columns", {})))
    st.caption("Profiling is computed locally; raw data never leaves this app.")

    columns = profiling.get("columns", {})
    if columns:
        df_columns = pd.DataFrame.from_dict(columns, orient="index").reset_index().rename(columns={"index": "column"})
        st.dataframe(df_columns)

    if issues:
        df_issues = pd.DataFrame(issues)
        st.dataframe(df_issues)
    else:
        st.success("No heuristic issues detected.")


def render_ai_analysis_tab():
    st.subheader("AI Analysis & Fix Suggestions")
    ai_analysis = st.session_state.get("ai_analysis")
    issues = st.session_state.get("issues", [])
    if not ai_analysis:
        st.info("Run an analysis to see AI insights.")
        return

    overall_summary = ai_analysis.get("overall_summary", "")
    if isinstance(overall_summary, dict):
        st.json(overall_summary)
    else:
        st.write(overall_summary)
    explanations = ai_analysis.get("issue_explanations", [])
    severity_lookup = {iss["issue_id"]: iss.get("severity") for iss in issues if iss.get("issue_id")}
    if explanations:
        st.markdown("### Issues & Explanations")
        for idx, item in enumerate(explanations, start=1):
            sev = severity_lookup.get(item.get("issue_id")) or item.get("severity") or ""
            color = "#ff6b6b" if sev == "HIGH" else "#f2b600" if sev == "MEDIUM" else "#3dd598" if sev == "LOW" else "#7ac7ff"
            issue_html = f"""
            <div class="dq-card" style="border-left: 6px solid {color};">
                <div class="dq-pill" style="background:{color}1a;border-color:{color};">#{idx}</div>
                <div class="dq-pill" style="background:{color}1a;border-color:{color};">ID: {item.get('issue_id','')}</div>
                <h4>{item.get('explanation','')}</h4>
            </div>
            """
            st.markdown(issue_html, unsafe_allow_html=True)
    elif issues:
        st.caption("AI did not return per-issue explanations; showing heuristic issues instead.")
        for idx, issue in enumerate(issues, start=1):
            sev = issue.get("severity", "")
            color = "#ff6b6b" if sev == "HIGH" else "#f2b600" if sev == "MEDIUM" else "#3dd598" if sev == "LOW" else "#7ac7ff"
            issue_html = f"""
            <div class="dq-card" style="border-left: 6px solid {color};">
                <div class="dq-pill" style="background:{color}1a;border-color:{color};">#{idx}</div>
                <div class="dq-pill" style="background:{color}1a;border-color:{color};">{issue['issue_type']}</div>
                <h4>{issue['details']}</h4>
            </div>
            """
            st.markdown(issue_html, unsafe_allow_html=True)

    sql_fixes = _stringify_fixes(ai_analysis.get("recommended_sql_fixes", ""))
    client = st.session_state.get("sb_client")
    if sql_fixes:
        st.markdown("**SQL Fixes**")
        st.code(sql_fixes, language="sql")
        if client:
            confirm = st.checkbox("I understand this will modify data in Supabase.", value=False)
            dangerous = any(tok in sql_fixes.lower() for tok in ["update", "delete", "insert", "alter"])
            if st.button("Apply SQL Fix", disabled=not confirm or not dangerous):
                try:
                    cleaned_sql = _clean_sql(sql_fixes)
                    if not cleaned_sql:
                        st.error("Could not parse SQL content. Please run it manually.")
                        return
                    # Attempt to run the SQL via PostgREST RPC (requires an 'exec_sql' function in Supabase).
                    res = client.postgrest.rpc("exec_sql", {"sql": cleaned_sql}).execute()
                    st.success("SQL fix applied. Verify your data in Supabase.")
                except Exception as exc:
                    st.error(f"Failed to apply SQL automatically. Please run manually in Supabase SQL editor. Error: {exc}")

    rules = ai_analysis.get("recommended_rules", [])
    if rules:
        st.markdown("**Suggested Validation Rules**")
        for rule in rules:
            st.markdown(f"- {rule}")

    python_fixes = _stringify_fixes(ai_analysis.get("recommended_python_fixes", ""))
    if python_fixes:
        st.markdown("**Python Fixes**")
        st.code(python_fixes, language="python")


def render_run_history_tab():
    st.subheader("Run History")
    runs = get_recent_runs(limit=20)
    if not runs:
        st.info("No stored runs yet.")
        return

    df_runs = pd.DataFrame(runs)
    st.dataframe(df_runs)

    selected_run_id = st.selectbox("View run details", [""] + [str(r["id"]) for r in runs], format_func=lambda x: x or "Select a run")
    if selected_run_id:
        run_id_int = int(selected_run_id)
        issues = get_issues_for_run(run_id_int)
        st.markdown(f"**Issues for run {run_id_int}**")
        st.dataframe(pd.DataFrame(issues))
