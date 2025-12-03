import html
import os
import smtplib
from email.mime.text import MIMEText
from typing import Dict, Any, List, Tuple


def _build_email_body(profiling: Dict, issues: List[Dict], ai_analysis: Dict) -> str:
    def esc(value: Any) -> str:
        return html.escape(str(value or ""))

    table_ref = esc(profiling.get("table_ref", ""))
    row_count = esc(profiling.get("row_count", 0))
    overall_summary = esc(ai_analysis.get("overall_summary", ""))
    sql_fixes = ai_analysis.get("recommended_sql_fixes")
    python_fixes = ai_analysis.get("recommended_python_fixes")
    issue_explanations = ai_analysis.get("issue_explanations", []) or []
    recommended_rules = ai_analysis.get("recommended_rules", []) or []

    severity_colors = {
        "high": "#d1434b",
        "medium": "#e67e22",
        "low": "#2c7be5",
    }

    issue_rows: List[str] = []
    severity_lookup = {issue.get("issue_id"): issue.get("severity", "") for issue in issues if issue.get("issue_id")}
    for idx, issue in enumerate(issues, start=1):
        severity = esc(issue.get("severity", ""))
        color = severity_colors.get(issue.get("severity", "").lower(), "#6c757d")
        issue_rows.append(
            f"""
            <tr>
                <td style="padding:8px 10px;border:1px solid #e9ecef;">{idx}</td>
                <td style="padding:8px 10px;border:1px solid #e9ecef;">{esc(issue.get('issue_type', ''))}</td>
                <td style="padding:8px 10px;border:1px solid #e9ecef;">
                    <span style="display:inline-block;padding:4px 8px;border-radius:999px;background:{color};color:#fff;font-size:12px;">{severity or 'N/A'}</span>
                </td>
                <td style="padding:8px 10px;border:1px solid #e9ecef;">{esc(issue.get('details', ''))}</td>
            </tr>
            """
        )

    issues_section = (
        "\n".join(issue_rows)
        if issue_rows
        else """
            <tr>
                <td colspan="4" style="padding:12px 10px;border:1px solid #e9ecef;text-align:center;color:#6c757d;">
                    No issues detected in this run.
                </td>
            </tr>
        """
    )

    sql_fix_block = (
        f"""
        <div style="margin-top:16px;">
            <div style="font-weight:600;margin-bottom:6px;color:#2c3e50;">Recommended SQL Fixes</div>
            <pre style="background:#0f172a;color:#e2e8f0;padding:12px;border-radius:10px;font-size:13px;line-height:1.5;white-space:pre-wrap;border:1px solid #1e293b;overflow:auto;">{esc(sql_fixes)}</pre>
        </div>
        """
        if sql_fixes
        else ""
    )

    python_fix_block = (
        f"""
        <div style="margin-top:16px;">
            <div style="font-weight:600;margin-bottom:6px;color:#2c3e50;">Recommended Python Fixes</div>
            <pre style="background:#0f172a;color:#e2e8f0;padding:12px;border-radius:10px;font-size:13px;line-height:1.5;white-space:pre-wrap;border:1px solid #1e293b;overflow:auto;">{esc(python_fixes)}</pre>
        </div>
        """
        if python_fixes
        else ""
    )

    explanation_cards = []
    for idx, item in enumerate(issue_explanations, start=1):
        sev = severity_lookup.get(item.get("issue_id")) or item.get("severity") or ""
        sev_color = severity_colors.get(str(sev).lower(), "#6c757d")
        explanation_cards.append(
            f"""
            <div style="border:1px solid #e9ecef;border-radius:12px;padding:12px 14px;margin-bottom:10px;background:#f8fafc;">
              <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;">
                <span style="display:inline-block;padding:4px 10px;border-radius:999px;background:{sev_color};color:#fff;font-size:12px;font-weight:600;">{esc(sev or 'N/A')}</span>
                <span style="font-weight:700;color:#111827;">Issue {esc(item.get('issue_id', idx))}</span>
              </div>
              <div style="color:#374151;line-height:1.5;">{esc(item.get('explanation', ''))}</div>
            </div>
            """
        )

    explanations_block = (
        "\n".join(explanation_cards)
        if explanation_cards
        else '<div style="color:#6c757d;font-style:italic;">No AI explanations returned.</div>'
    )

    rules_list = (
        "<ul style='margin:8px 0 0 16px;color:#374151;line-height:1.6;'>"
        + "".join(f"<li>{esc(rule)}</li>" for rule in recommended_rules)
        + "</ul>"
        if recommended_rules
        else '<div style="color:#6c757d;font-style:italic;">No additional rules suggested.</div>'
    )

    return f"""
    <html>
      <body style="margin:0;padding:0;font-family:'Segoe UI', -apple-system, BlinkMacSystemFont, 'Helvetica Neue', Arial, sans-serif;background:#f6f8fb;color:#1f2933;">
        <div style="max-width:720px;margin:24px auto;padding:24px;background:#ffffff;border:1px solid #e8ecf3;border-radius:14px;box-shadow:0 10px 25px rgba(15,23,42,0.08);">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;">
            <div style="font-size:22px;font-weight:700;color:#111827;">Data Quality Run Summary</div>
            <div style="padding:6px 12px;border-radius:999px;background:#eef2ff;color:#4338ca;font-weight:600;font-size:12px;letter-spacing:0.3px;">Automated Report</div>
          </div>
          <p style="margin:0 0 16px;color:#4b5563;">Here are the highlights from the latest data quality checks.</p>

          <div style="display:flex;gap:12px;margin-bottom:18px;flex-wrap:wrap;">
            <div style="flex:1;min-width:220px;padding:14px 16px;border:1px solid #e9ecef;border-radius:12px;background:#f8fafc;">
              <div style="font-size:12px;text-transform:uppercase;color:#6c757d;letter-spacing:0.5px;">Table</div>
              <div style="font-size:18px;font-weight:700;color:#111827;margin-top:4px;">{table_ref or 'N/A'}</div>
            </div>
            <div style="flex:1;min-width:220px;padding:14px 16px;border:1px solid #e9ecef;border-radius:12px;background:#f8fafc;">
              <div style="font-size:12px;text-transform:uppercase;color:#6c757d;letter-spacing:0.5px;">Row Count</div>
              <div style="font-size:18px;font-weight:700;color:#111827;margin-top:4px;">{row_count}</div>
            </div>
          </div>

          <div style="margin-bottom:18px;">
            <div style="font-weight:700;font-size:16px;color:#111827;margin-bottom:8px;">AI Summary</div>
            <div style="padding:12px 14px;border:1px solid #e9ecef;border-radius:12px;background:#f8fafc;color:#374151;line-height:1.6;">{overall_summary or 'No summary available.'}</div>
          </div>

          <div style="margin-bottom:18px;">
            <div style="font-weight:700;font-size:16px;color:#111827;margin-bottom:6px;">AI Analysis & Fix Suggestions</div>
            {explanations_block}
          </div>

          <div style="margin-bottom:10px;font-weight:700;font-size:16px;color:#111827;">Detected Issues</div>
          <table style="width:100%;border-collapse:collapse;border:1px solid #e9ecef;border-radius:12px;overflow:hidden;">
            <thead>
              <tr style="background:#f1f5f9;color:#111827;text-align:left;">
                <th style="padding:10px 12px;border:1px solid #e9ecef;font-size:13px;">#</th>
                <th style="padding:10px 12px;border:1px solid #e9ecef;font-size:13px;">Issue</th>
                <th style="padding:10px 12px;border:1px solid #e9ecef;font-size:13px;">Severity</th>
                <th style="padding:10px 12px;border:1px solid #e9ecef;font-size:13px;">Details</th>
              </tr>
            </thead>
            <tbody>
              {issues_section}
            </tbody>
          </table>

          {sql_fix_block}
          {python_fix_block}

          <div style="margin-top:16px;">
            <div style="font-weight:600;margin-bottom:6px;color:#2c3e50;">Suggested Data Quality Rules</div>
            {rules_list}
          </div>

          <div style="margin-top:24px;padding-top:14px;border-top:1px solid #e9ecef;color:#6c757d;font-size:12px;line-height:1.6;">
            This report was generated automatically by the Data Quality Agent. For questions, reply to this email.
          </div>
        </div>
      </body>
    </html>
    """


def send_run_email(profiling: Dict, issues: List[Dict], ai_analysis: Dict) -> Tuple[bool, str]:
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = os.environ.get("SMTP_PORT")
    smtp_user = os.environ.get("SMTP_USER")
    smtp_pass = os.environ.get("SMTP_PASS")
    smtp_security = (os.environ.get("SMTP_SECURITY") or "ssl").lower()  # "ssl" or "starttls"
    to_addr = os.environ.get("ALERT_EMAIL_TO")
    from_addr = os.environ.get("ALERT_EMAIL_FROM", smtp_user)

    if not all([smtp_host, smtp_port, smtp_user, smtp_pass, to_addr, from_addr]):
        return False, "Missing SMTP configuration"

    body = _build_email_body(profiling, issues, ai_analysis)
    msg = MIMEText(body, "html")
    msg["Subject"] = f"DQ Run - {profiling.get('table_ref', '')}"
    msg["From"] = from_addr
    msg["To"] = to_addr

    try:
        port_int = int(smtp_port)
        if smtp_security == "starttls":
            with smtplib.SMTP(smtp_host, port_int) as server:
                server.starttls()
                server.login(smtp_user, smtp_pass)
                server.sendmail(from_addr, [to_addr], msg.as_string())
        else:
            with smtplib.SMTP_SSL(smtp_host, port_int) as server:
                server.login(smtp_user, smtp_pass)
                server.sendmail(from_addr, [to_addr], msg.as_string())
        return True, "Email sent"
    except Exception as exc:
        return False, f"Email failed: {exc}"
