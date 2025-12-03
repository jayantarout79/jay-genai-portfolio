# AI Data Quality Assistant (Supabase)

Streamlit application that profiles Supabase (Postgres) tables, runs heuristic data-quality checks, and asks an OpenAI model for targeted explanations and fixes—without ever sending raw data outside the app. Run history is persisted in Supabase. Built to be privacy-first and GitHub-ready (no secrets committed).

## Architecture
- **Supabase ➜ Profiling:** Sample limited rows to compute column stats and table-level metrics locally.
- **Rule Engine:** Metadata-only heuristics (nulls, cardinality, negative/zero amounts, future dates, duplicates).
- **AI Agent:** Builds a prompt from profiling metadata + issue summaries and calls OpenAI once for grouped root causes and fix suggestions.
- **UI:** Streamlit tabs for running checks, viewing profiling/issue tables, AI analysis, and run history.
- **Storage:** Supabase tables `dq_runs` and `dq_issues` store run metadata.

## Data Privacy
The AI agent operates only on aggregated profiling metadata and data-quality issue summaries. Raw business data and PII are never sent to the model.

## Quickstart
1) Python 3.10+. Create/activate a virtualenv.
2) Install dependencies:
```bash
pip install -r requirements.txt
```
3) Copy environment template and fill in your secrets (do **not** commit your real values):
```bash
cp .env.example .env
# edit .env with your Supabase/OpenAI/SMTP details
```
4) Run the app:
```bash
streamlit run app.py
```

## Supabase DDL
Create the run-history tables in your Supabase PostgreSQL database:
```sql
CREATE TABLE public.dq_runs (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  table_ref text NOT NULL,
  run_timestamp timestamptz NOT NULL DEFAULT now(),
  row_count integer,
  issue_count integer
);

CREATE TABLE public.dq_issues (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  run_id bigint NOT NULL REFERENCES public.dq_runs(id) ON DELETE CASCADE,
  column_name text,
  issue_type text,
  severity text,
  rows_affected integer,
  details text
);

CREATE INDEX dq_issues_run_id_idx ON public.dq_issues(run_id);
```

## Usage
- Use the sidebar to choose schema (default `public`) and select or type a table, then click **Run Data Quality Analysis**.
- Review **Profile & Issues** for column metrics and heuristic findings.
- Check **AI Analysis & Fix Suggestions** for grouped explanations plus SQL/Python fixes.
- See **Run History** for recent runs stored in Supabase.

## Configuration
Set the following in `.env` or Streamlit secrets:
- `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY` (preferred) or `NEXT_PUBLIC_SUPABASE_ANON_KEY`
- `OPENAI_API_KEY` (optional but recommended for AI insights)
- `SMTP_HOST`, `SMTP_PORT`, `SMTP_USER`, `SMTP_PASS`, `ALERT_EMAIL_FROM`, `ALERT_EMAIL_TO`, `SMTP_SECURITY` (`ssl` or `starttls`) for email notifications

## Deployment Notes
- `.env` is ignored by git; publish only the sanitized `.env.example`.
- Streamlit Cloud: add the same environment variables in the dashboard; Supabase tables must already exist.
- The SQL auto-fix button is guarded with an explicit checkbox and basic mutation detection; review SQL before applying.
