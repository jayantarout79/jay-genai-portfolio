"""Central color palette, typography, and layout helpers."""

from __future__ import annotations

import streamlit as st

LIGHT_PALETTE = {
    "background": "#0e1626",
    "surface": "rgba(18,30,48,0.9)",
    "surface_alt": "rgba(26,42,68,0.85)",
    "border": "rgba(255,255,255,0.08)",
    "text": "#eef4ff",
    "muted": "#9fb3d2",
    "accent": "#3b82f6",
    "accent_soft": "rgba(59,130,246,0.35)",
    "positive": "#34d399",
    "negative": "#f87171",
    "neutral": "#9aa5b1",
}

DARK_PALETTE = {
    "background": "#0e1119",
    "surface": "#151a24",
    "surface_alt": "#1f2532",
    "border": "#242c3c",
    "text": "#f3f6fb",
    "muted": "#9aa5b1",
    "accent": "#5ea0ff",
    "accent_soft": "#1f3b62",
    "positive": "#34d399",
    "negative": "#f87171",
    "neutral": "#cbd5f5",
}

FONT_FAMILY = "'SF Pro Display', 'Segoe UI', system-ui, -apple-system, BlinkMacSystemFont, sans-serif"
CURRENT_PALETTE = LIGHT_PALETTE.copy()


def _palette_css(prefix: str, palette: dict) -> str:
    return "\n".join(f"{prefix}-{key}: {value};" for key, value in palette.items())


def inject_global_styles(dark: bool = False) -> None:
    """Inject a cohesive global theme with cards and typography."""
    global CURRENT_PALETTE
    palette = DARK_PALETTE if dark else LIGHT_PALETTE
    CURRENT_PALETTE = palette
    st.markdown(
        f"""
        <style>
        :root {{
            {_palette_css('--', palette)}
            --font-family: {FONT_FAMILY};
            font-size: 14px;
        }}
        * {{
            font-family: var(--font-family);
        }}
        body, [data-testid="stAppViewContainer"] {{
            background: radial-gradient(circle at 20% 20%, rgba(59,130,246,0.1), rgba(11,21,39,1) 35%), linear-gradient(180deg, #0b1527 0%, #0f1f3a 50%, #0b1527 100%);
            color: var(--text-color);
        }}
        .card {{
            background: var(--surface);
            border-radius: 14px;
            padding: 0.95rem;
            border: 1px solid var(--border);
            box-shadow: 0 20px 40px rgba(0,0,0,0.25);
        }}
        .nav-card {{
            background: linear-gradient(135deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
            border-radius: 14px;
            padding: 0.65rem;
            border: 1px solid var(--border);
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        }}
        .nav-card .stRadio > div {{
            gap: 0.2rem;
        }}
        .nav-card label {{
            width: 100%;
            background: rgba(255,255,255,0.03);
            border-radius: 14px;
            padding: 0.5rem 0.75rem;
            border: 1px solid transparent;
            color: var(--text-color);
            font-weight: 600;
            font-size: 0.95rem;
        }}
        .nav-card label[data-testid="stMarkdownContainer"] {{
            padding: 0;
        }}
        .nav-card .stRadio div[role="radiogroup"] > label:hover {{
            border-color: rgba(255,255,255,0.12);
        }}
        .nav-card input[checked] + div + label {{
            background: linear-gradient(135deg, rgba(59,130,246,0.35), rgba(59,130,246,0.15));
            color: #e6efff;
            border-color: rgba(59,130,246,0.6);
        }}
        .stButton>button, .stDownloadButton>button {{
            border-radius: 999px;
            border: 1px solid rgba(255,255,255,0.1);
            padding: 0.6rem 1.1rem;
            font-weight: 600;
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            color: #e8f1ff;
            box-shadow: 0 10px 30px rgba(59,130,246,0.35);
            font-size: 0.95rem;
        }}
        .stButton>button:disabled {{
            opacity: 0.35 !important;
            cursor: not-allowed;
            box-shadow: none;
        }}
        .stButton>button[kind="secondary"] {{
            background: rgba(255,255,255,0.06);
            color: var(--text-color);
            border: 1px solid var(--border);
            box-shadow: none;
        }}
        .stButton>button:focus:not(:active) {{
            box-shadow: 0 0 0 3px var(--accent_soft);
        }}
        .metric-pill {{
            background: var(--surface);
            border-radius: 18px;
            padding: 0.85rem 1.1rem;
            border: 1px solid var(--border);
        }}
        .metric-pill span {{
            display: block;
        }}
        .metric-pill .label {{
            text-transform: uppercase;
            font-size: 0.72rem;
            color: var(--muted-color);
            letter-spacing: 0.06em;
        }}
        .metric-pill .value {{
            font-size: 1.44rem;
            font-weight: 600;
            color: var(--text-color);
        }}
        .chat-bubble {{
            border-radius: 16px;
            padding: 0.85rem 1rem;
            border: 1px solid var(--border);
            background: var(--surface);
            box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
        }}
        .chat-bubble.assistant {{
            background: #eef4ff;
            border-color: #d9e4ff;
        }}
        .chat-bubble.user {{
            background: #fff1f0;
            border-color: #ffd4cf;
        }}
        .sticky-toolbar {{
            position: sticky;
            top: 0;
            z-index: 5;
            background: var(--surface);
            padding-bottom: 0.8rem;
            margin-bottom: 0.8rem;
            border-bottom: 1px solid var(--border);
        }}
        .processing-banner {{
            display: flex;
            align-items: center;
            gap: 0.9rem;
            padding: 0.95rem 1.1rem;
            border-radius: 16px;
            border: 1px solid var(--border);
            background: linear-gradient(135deg, rgba(59,130,246,0.2), rgba(255,255,255,0.04));
            margin-bottom: 0.6rem;
        }}
        .processing-dot {{
            width: 14px;
            height: 14px;
            border-radius: 999px;
            background: var(--accent-color);
            box-shadow: 0 0 0 rgba(37, 99, 235, 0.4);
            animation: pulse 1.4s ease-in-out infinite;
        }}
        @keyframes pulse {{
            0% {{ box-shadow: 0 0 0 0 rgba(37, 99, 235, 0.5); }}
            70% {{ box-shadow: 0 0 0 16px rgba(37, 99, 235, 0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(37, 99, 235, 0); }}
        }}
        .gradient-card {{
            background: linear-gradient(135deg, rgba(37,99,235,0.09), rgba(79,70,229,0.08));
            border: 1px solid var(--accent-soft);
        }}
        .status-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 0.35rem;
            margin-bottom: 0.6rem;
        }}
        .status-row {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
            gap: 0.35rem;
            margin-bottom: 0.6rem;
        }}
        .status-pill {{
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            padding: 0.65rem 0.75rem;
            display: flex;
            align-items: center;
            gap: 0.55rem;
            background: linear-gradient(145deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03));
            box-shadow: inset 0 1px 0 rgba(255,255,255,0.08);
        }}
        .status-pill .dot {{
            width: 10px;
            height: 10px;
            border-radius: 999px;
        }}
        .status-pill .label {{
            font-weight: 600;
            color: #e9f2ff;
            letter-spacing: 0.01em;
            font-size: 0.95rem;
        }}
        .status-pill .sub {{
            color: var(--muted-color);
            font-size: 0.82rem;
        }}
        .upload-chips {{
            display: flex;
            flex-wrap: wrap;
            gap: 0.35rem;
            margin-top: 0.35rem;
        }}
        .upload-chips .chip {{
            padding: 0.3rem 0.55rem;
            border-radius: 12px;
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.08);
            font-size: 0.82rem;
            color: #e6efff;
        }}
        .info-chip {{
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            padding: 0.35rem 0.65rem;
            border-radius: 12px;
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.1);
            font-size: 0.85rem;
            color: #e6efff;
            margin-bottom: 0.4rem;
        }}
        /* Hide default file uploader instructional text, keep button */
        div[data-testid="stFileUploadDropzone"] * {{display:none !important;}}
        div[data-testid="stFileUploadDropzone"] {{
            padding:0.4rem !important;
            min-height:0 !important;
            background: rgba(255,255,255,0.04);
            border: 1px dashed rgba(255,255,255,0.08);
            position: relative;
        }}
        div[data-testid="stFileUploader"] label,
        div[data-testid="stFileUploader"] p,
        div[data-testid="stFileUploader"] span,
        div[data-testid="stFileUploader"] small,
        div[data-testid="stFileUploadInstructions"],
        div[data-testid="stFileUploadDropzoneInstructions"],
        div[data-testid="stFileUploadDropzone"] span {{
            display: none !important;
        }}
        div[data-testid="stFileUploader"] ul {{
            display: none !important;  /* hide uploaded file list in sidebar */
        }}
        div[data-testid="stFileUploadDropzone"]::before {{
            content: "Browse files";
            display: inline-block;
            padding: 0.45rem 0.9rem;
            background: linear-gradient(135deg, #3b82f6, #2563eb);
            color: #e8f1ff;
            border-radius: 10px;
            font-weight: 600;
            font-size: 0.95rem;
            border: 1px solid rgba(255,255,255,0.12);
            box-shadow: 0 8px 20px rgba(59,130,246,0.25);
            cursor: pointer;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def sentiment_color(label: str) -> str:
    """Return palette color for a sentiment label."""
    mapping = {
        "positive": CURRENT_PALETTE["positive"],
        "negative": CURRENT_PALETTE["negative"],
        "neutral": CURRENT_PALETTE["neutral"],
    }
    return mapping.get(label, CURRENT_PALETTE["muted"])
