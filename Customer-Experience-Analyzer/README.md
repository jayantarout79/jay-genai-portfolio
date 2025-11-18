# Customer Experience Analyzer

A Streamlit dashboard for exploring Net Promoter Score (NPS) survey data, surfacing AI-generated insights, and chatting with your metrics. The app bundles ingestion, validation, KPI tracking, AI theming, comparison views, and an executive-summary writer into a single interface.

---

## 1. Getting Started

1. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
2. **(Optional) Generate demo data** – produces `data/sample_survey.csv` with 50k realistic records.
   ```bash
   python generate_sample_survey_2024.py
   ```
3. **Provide OpenAI credentials**
   - Add `OPENAI_API_KEY` (and optional `OPENAI_MODEL`) to a `.env` file, or
   - Paste your key into the sidebar input once the UI loads (stored only in `st.session_state` for the active browser session; see `app.py:262`).
4. **Run the dashboard**
   ```bash
   streamlit run app.py
   ```

---

## 2. Codebase Tour

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit entry point: sets up theming, sidebar controls, page tabs, and wires session state. |
| `ui/layout.py` | All tab renderers and shared UI helpers (filters, caching, keyword cloud). |
| `ui/state.py` | Declares the `st.session_state` keys used across tabs for caching data, API keys, and chat history. |
| `core/data_loader.py` | CSV ingestion, validation, profiling, data-health checks, filtering utilities. |
| `core/metrics.py` & `core/visuals.py` | NPS math, comparison helpers, keyword extraction, plus Plotly chart builders. |
| `ai/llm_client.py` | Thin wrapper around the OpenAI Responses API with offline fallbacks if the key/library is missing. |
| `ai/theme_extractor.py`, `ai/summary_generator.py`, `ai/chatbot.py` | Feature-specific LLM orchestrators (themes, exec summary, chatbot). |
| `data/sample_survey.csv` | Demo dataset consumed by default via `app.DATA_PATH`. |
| `generate_sample_survey_2024.py` | Utility to regenerate or customize the demo CSV. |

---

## 3. Data & Processing Pipeline

1. **Ingestion & validation**  
   `core/data_loader.prepare_survey_dataframe` enforces required columns (`response_id`, `date`, `channel`, etc.), coerces types, trims whitespace, and adds defaults for optional fields (`country`, `store_id`).
2. **Storage & session state**  
   Cleaned DataFrames are stored in `st.session_state["dataframe"]` via `ui.state.set_dataframe`, keeping user-uploaded data isolated per session.
3. **Filtering & caching**  
   Shared helpers in `ui/layout` (`_filter_controls`, `_get_metrics_from_cache`, `_get_themes`) apply date/channel/region filters and cache expensive computations keyed by the filter fingerprint for snappy tab switching.
4. **Metrics computation**  
   `core/metrics` exposes reusable functions for KPIs, trends, dimension splits, and comparisons that power KPI cards and visuals.
5. **Visualization**  
   `core/visuals` turns DataFrames into Plotly figures (trend lines, grouped bars) which Streamlit renders responsively.
6. **AI augmentation**  
   When an OpenAI key is configured, `ai/llm_client.LLMClient` powers:
   - Theme extraction from comments with JSON parsing and heuristic fallback.
   - Executive summaries that mix metric context with comparison deltas.
   - Chatbot answers grounded in metrics, filters, and cached themes.
   Without a key, deterministic offline strings keep the UI functional.

---

## 4. UI Walkthrough

Each Streamlit tab lives in `ui/layout.py`, making it easy to locate related logic.

### Upload & Health (`render_upload_tab`)
- Upload a CSV or load the demo dataset (`load_demo_dataset` → `core/data_loader.load_survey_data`).
- Displays quick metrics: row count, available date span, unique channels/regions, data-health stats, and a preview table.
- Uploading replaces the current session dataset and flushes cached metrics/themes.

### NPS Overview (`render_nps_tab`)
- Filter controls for time frame, channels, and regions.
- KPI deck (overall NPS, promoter/passive/detractor %, total responses) backed by `core.metrics.compute_kpis`.
- Trend line plus NPS-by-channel/region bar charts (`core.visuals`).
- Keyword cloud derived from filtered comments (`core.metrics.top_keywords`).
- Metrics bundle is cached per filter combo to avoid recomputation.

### AI Themes (`render_themes_tab`)
- Shares the filter controls but focuses on qualitative insights.
- Clicking “Show Themes” triggers `ai.theme_extractor.extract_themes`, which:
  1. Sends up to 250 filtered comments to the LLM requesting JSON.
  2. Validates/cleans the payload and enriches it with sentiment + volume percentages.
  3. Falls back to deterministic heuristics if JSON parsing or the API fails.
- Themes are displayed in paired cards with sentiment badges and example quotes.

### What Changed? (`render_comparison_tab`)
- Allows two distinct date selections plus channel/region filters.
- Computes KPIs for period A vs. period B via `core.metrics.comparison_nps`.
- Shows delta KPIs, grouped comparison bars, and a tabular breakdown (responses + ΔNPS) powered by `core.metrics.comparison_table`.
- Cached results feed the Exec Summary and chatbot context.

### Exec Summary (`render_summary_tab`)
- Lets users define a time window (presets or custom) and filters before generating an executive briefing.
- Reuses cached themes (limited to five) and comparison context to build the prompt.
- `ai.summary_generator.generate_executive_summary` crafts a <=200-word narrative and proposes two focus actions; offline fallback stitches together key stats if the LLM is unavailable.

### Chatbot (`render_chat_tab`)
- Conversational interface referencing the filtered dataset, cached KPIs, and last themes.
- `ai.chatbot.answer_question` infers simple filters from text (channels/regions mentioned), recomputes KPIs if necessary, and builds a compact JSON context for the LLM.
- Responses are appended to `st.session_state["chat_history"]`, giving conversational continuity.

---

## 5. How the Sidebar Works

- Theme + layout CSS is injected via `_inject_dashboard_theme` for a polished dashboard feel.
- Sidebar card introduces the app and provides the OpenAI key field. Editing the key updates `st.session_state["api_key_override"]`, which overrides `.env` values for the rest of the session.
- Displays the active data source (uploaded filename vs. demo) so users know which dataset powers the tabs.

---

## 6. Extending the App

- **New dimensions or filters**: ensure the column exists in the cleaned DataFrame, then extend `_filter_controls`, KPI cards, and `core.metrics` functions as needed.
- **Additional AI workflows**: create a new module under `ai/`, wire it through `LLMClient`, and surface it in a tab via `ui/layout`.
- **Alternative visualizations**: add builders to `core/visuals.py` and call them in the relevant tab.
- **Persisting API keys/user settings**: hook into Streamlit’s secrets management or store encrypted values server-side; currently keys are intentionally session-scoped only.

---

## 7. Troubleshooting

- **“Missing required columns”** – ensure uploads contain the schema from `core/data_loader.REQUIRED_COLUMNS`.
- **No AI outputs** – confirm the OpenAI package is installed (`pip install openai`) and a valid key is provided; otherwise the UI will show deterministic placeholder text.
- **Cache confusion after uploads** – switching datasets auto-resets caches, but you can also manually clear by using “Rerun” in Streamlit if needed.

With these pieces, new contributors and stakeholders can understand how the Customer Experience Analyzer ingests data, computes KPIs, layers on AI insights, and powers each UI tab.
