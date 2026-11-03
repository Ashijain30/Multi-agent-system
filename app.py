import logging
import time
import os
import streamlit as st
from dotenv import load_dotenv

from pipeline import run_research_pipeline, validate_topic
from tools import sanitize_error_message, get_credential
from models import SourceItem, ResearchReport, CriticEvaluation

load_dotenv()

logger = logging.getLogger("researchmind.app")

# Theme preference persists across Streamlit reruns.
if "researchmind_theme" not in st.session_state:
    st.session_state.researchmind_theme = "Dark"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind · AI Research Agent",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,300&display=swap');

/* ── Reset & base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #e8e4dc;
}

.stApp {
    background: #0a0a0f;
    background-image:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(255,140,50,0.12) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255,80,30,0.08) 0%, transparent 55%);
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 3rem 4rem; max-width: 1200px; }

/* ── Hero header ── */
.hero {
    text-align: center;
    padding: 3.5rem 0 2.5rem;
    position: relative;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #ff8c32;
    margin-bottom: 1rem;
    opacity: 0.9;
}
.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 800;
    line-height: 1.0;
    letter-spacing: -0.03em;
    color: #f0ebe0;
    margin: 0 0 1rem;
}
.hero h1 span {
    color: #ff8c32;
}
.hero-sub {
    font-size: 1.05rem;
    font-weight: 300;
    color: #a09890;
    max-width: 580px;
    margin: 0 auto;
    line-height: 1.65;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(255,140,50,0.3), transparent);
    margin: 2rem 0;
}

/* ── Input card ── */
.input-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,140,50,0.15);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    backdrop-filter: blur(8px);
}

/* ── Streamlit input overrides ── */
.stTextInput > div > div > input {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(255,140,50,0.25) !important;
    border-radius: 10px !important;
    color: #f0ebe0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus {
    border-color: #ff8c32 !important;
    box-shadow: 0 0 0 3px rgba(255,140,50,0.12) !important;
}
.stTextInput > label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
    color: #ff8c32 !important;
    font-weight: 500 !important;
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #ff8c32 0%, #ff5a1a 100%) !important;
    color: #0a0a0f !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 2.2rem !important;
    cursor: pointer !important;
    transition: transform 0.15s, box-shadow 0.15s, opacity 0.15s !important;
    box-shadow: 0 4px 20px rgba(255,140,50,0.3) !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(255,140,50,0.4) !important;
    opacity: 0.95 !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* ── Pipeline step cards ── */
.step-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 14px;
    padding: 1.3rem 1.6rem;
    margin-bottom: 1rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.3s;
}
.step-card.active {
    border-color: rgba(255,140,50,0.4);
    background: rgba(255,140,50,0.04);
}
.step-card.done {
    border-color: rgba(80,200,120,0.3);
    background: rgba(80,200,120,0.03);
}
.step-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    border-radius: 14px 0 0 14px;
    background: rgba(255,255,255,0.05);
    transition: background 0.3s;
}
.step-card.active::before { background: #ff8c32; }
.step-card.done::before   { background: #50c878; }

.step-header {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    margin-bottom: 0.2rem;
}
.step-num {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    font-weight: 500;
    letter-spacing: 0.15em;
    color: #ff8c32;
    opacity: 0.7;
}
.step-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #f0ebe0;
}
.step-status {
    margin-left: auto;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.1em;
}
.status-waiting  { color: #555; }
.status-running  { color: #ff8c32; }
.status-done     { color: #50c878; }

/* ── Sources and Results panels ── */
.source-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.08em;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    margin-right: 0.5rem;
    text-transform: uppercase;
}
.source-badge.scraped {
    background: rgba(80,200,120,0.15);
    color: #50c878;
    border: 1px solid rgba(80,200,120,0.3);
}
.source-badge.fallback {
    background: rgba(255,140,50,0.15);
    color: #ff8c32;
    border: 1px solid rgba(255,140,50,0.3);
}

.source-item-card {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.8rem;
}
.source-title {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    font-size: 0.95rem;
    color: #f0ebe0;
    margin-bottom: 0.3rem;
}
.source-url {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #ff8c32;
    text-decoration: none;
    word-break: break-all;
}
.source-snippet {
    font-size: 0.85rem;
    color: #a09890;
    margin-top: 0.4rem;
    line-height: 1.5;
}

/* ── Report & Feedback Panels ── */
.report-panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,140,50,0.25);
    border-radius: 16px;
    padding: 2.2rem 2.6rem;
    margin-top: 1rem;
    line-height: 1.8;
}
.feedback-panel {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(80,200,120,0.25);
    border-radius: 16px;
    padding: 2.2rem 2.6rem;
    margin-top: 1rem;
    line-height: 1.7;
}
.panel-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
    padding-bottom: 0.7rem;
}
.panel-label.orange {
    color: #ff8c32;
    border-bottom: 1px solid rgba(255,140,50,0.15);
}
.panel-label.green {
    color: #50c878;
    border-bottom: 1px solid rgba(80,200,120,0.15);
}

.score-banner {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    padding: 0.8rem 1.4rem;
    border-radius: 10px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    margin-bottom: 1.2rem;
}
.score-pill {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.4rem;
    color: #50c878;
}

/* ── Section heading ── */
.section-heading {
    font-family: 'Syne', sans-serif;
    font-size: 1.3rem;
    font-weight: 700;
    color: #f0ebe0;
    margin: 2rem 0 1rem;
}

/* ── Toast-style notice ── */
.notice {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #605850;
    text-align: center;
    margin-top: 3rem;
    letter-spacing: 0.08em;
}
</style>
""", unsafe_allow_html=True)

# ── Light / dark theme ────────────────────────────────────────────────────────
theme_col_left, theme_col_right = st.columns([5, 1])
with theme_col_right:
    st.radio(
        "Appearance",
        options=["Dark", "Light"],
        horizontal=True,
        key="researchmind_theme",
        label_visibility="collapsed",
    )

if st.session_state.researchmind_theme == "Light":
    st.markdown("""
    <style>
    .stApp { background: #f5f6fa !important; background-image: none !important; color: #202536 !important; }
    html, body, [class*="css"] { color: #202536 !important; }
    .hero h1, .step-title, .section-heading, .source-title { color: #202536 !important; }
    .hero-sub, .source-snippet { color: #626b7b !important; }
    .input-card, .step-card, .report-panel, .feedback-panel, .source-item-card, .score-banner {
        background: rgba(255,255,255,0.92) !important;
        border-color: rgba(45,55,72,0.14) !important;
        box-shadow: 0 8px 24px rgba(25,35,55,0.045);
    }
    .stTextInput > div > div > input {
        background: #ffffff !important; color: #202536 !important;
        border-color: #d5dbe5 !important;
    }
    .notice { color: #7b8494 !important; }
    details summary { color: #626b7b !important; }
    .stRadio label, .stCaption, [data-testid="stCaptionContainer"] { color: #4b5568 !important; }
    </style>
    """, unsafe_allow_html=True)


# ── Helper: render a step card ────────────────────────────────────────────────
def step_card(num: str, title: str, state: str, desc: str = ""):
    status_map = {
        "waiting": ("WAITING", "status-waiting"),
        "running": ("● RUNNING", "status-running"),
        "done":    ("✓ DONE",   "status-done"),
    }
    label, cls = status_map.get(state, ("", ""))
    card_cls = {"running": "active", "done": "done"}.get(state, "")
    st.markdown(f"""
    <div class="step-card {card_cls}">
        <div class="step-header">
            <span class="step-num">{num}</span>
            <span class="step-title">{title}</span>
            <span class="step-status {cls}">{label}</span>
        </div>
        {"<div style='font-size:0.8rem;color:#706860;margin-top:0.25rem;'>"+desc+"</div>" if desc else ""}
    </div>
    """, unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
for key in ("results", "running", "done", "status_msg"):
    if key not in st.session_state:
        st.session_state[key] = {} if key == "results" else ("" if key == "status_msg" else False)


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-eyebrow">Multi-Agent AI System · Powered by Groq</div>
    <h1>Research<span>Mind</span></h1>
    <p class="hero-sub">
        Four specialized AI agents collaborate — searching, multi-source reading,
        writing, and iterative critic refinement — to deliver an authoritative research report.
    </p>
</div>
<div class="divider"></div>
""", unsafe_allow_html=True)

# Model & Provider status indicators (checked via unified get_credential)
groq_model_name = get_credential("GROQ_MODEL", "openai/gpt-oss-120b")
groq_has_key = bool(get_credential("GROQ_API_KEY"))
tavily_has_key = bool(get_credential("TAVILY_API_KEY"))

model_col, key_col = st.columns([3, 2])
with model_col:
    st.caption(f"⚡ LLM provider: **Groq** · Model configured: `{groq_model_name}`")
with key_col:
    if groq_has_key and tavily_has_key:
        st.caption("🔐 API keys active (Groq + Tavily)")
    elif not groq_has_key and not tavily_has_key:
        st.caption("⚠️ Missing `GROQ_API_KEY` and `TAVILY_API_KEY` in environment or secrets")
    elif not groq_has_key:
        st.caption("⚠️ Missing `GROQ_API_KEY`")
    else:
        st.caption("⚠️ Missing `TAVILY_API_KEY`")

# ── Layout: input left, pipeline right ───────────────────────────────────────
col_input, col_spacer, col_pipeline = st.columns([5, 0.5, 4])

with col_input:
    st.markdown('<div class="input-card">', unsafe_allow_html=True)
    topic = st.text_input(
        "Research Topic",
        placeholder="e.g. CRISPR gene editing clinical breakthroughs in 2025",
        key="topic_input",
        label_visibility="visible",
    )
    run_btn = st.button("⚡  Run Research Pipeline", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Example chips
    st.markdown("""
    <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-bottom:1.5rem;">
        <span style="font-family:'DM Mono',monospace;font-size:0.68rem;color:#605850;letter-spacing:0.1em;">TRY →</span>
    """, unsafe_allow_html=True)
    examples = [
        "CRISPR gene editing breakthroughs in 2025",
        "Quantum computing error mitigation architectures",
        "Autonomous AI agents in software engineering",
    ]
    for ex in examples:
        st.markdown(f"""
        <span style="
            background:rgba(255,255,255,0.04);
            border:1px solid rgba(255,255,255,0.08);
            border-radius:6px;
            padding:0.25rem 0.7rem;
            font-size:0.75rem;
            color:#a09890;
            font-family:'DM Sans',sans-serif;
            cursor:default;
        ">{ex}</span>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_pipeline:
    st.markdown('<div class="section-heading">Agent Pipeline</div>', unsafe_allow_html=True)

    r = st.session_state.results

    def s(step: str) -> str:
        if not r:
            return "waiting"
        steps = ["search", "reader", "writer", "critic"]
        if step in r and r[step]:
            return "done"
        if st.session_state.running:
            for k in steps:
                if k not in r or not r[k]:
                    return "running" if k == step else "waiting"
        return "waiting"

    step_card("01", "Search Agent",  s("search"), "Discovers multi-source web references")
    step_card("02", "Reader Agent",  s("reader"), "Scrapes 3–5 unique URLs with fallback")
    step_card("03", "Writer Agent",  s("writer"), "Synthesizes 6 verified report sections")
    step_card("04", "Critic Agent",  s("critic"), "Audits evidence & refines (max 2 loops)")


# ── Run pipeline ──────────────────────────────────────────────────────────────
if run_btn:
    try:
        clean_topic = validate_topic(topic)
    except ValueError as ve:
        st.warning(str(ve))
        clean_topic = None

    if clean_topic:
        missing_keys = []
        if not get_credential("GROQ_API_KEY"):
            missing_keys.append("GROQ_API_KEY")
        if not get_credential("TAVILY_API_KEY"):
            missing_keys.append("TAVILY_API_KEY")

        if missing_keys:
            st.error(f"Cannot run pipeline: missing required API key(s): {', '.join(missing_keys)}. Please add them to `.env` or Streamlit secrets.")
        else:
            st.session_state.results = {}
            st.session_state.running = True
            st.session_state.done = False
            st.rerun()

if st.session_state.running and not st.session_state.done:
    topic_val = st.session_state.topic_input
    status_placeholder = st.empty()

    def on_step_start(step_key: str, step_title: str, description: str):
        icons = {
            "search": "🔍",
            "reader": "📄",
            "writer": "✍️",
            "critic": "🧐",
        }
        icon = icons.get(step_key, "⚡")
        status_placeholder.info(f"{icon} **{step_title}**: {description}")

    def on_step_complete(step_key: str, output: str, current_state: dict):
        st.session_state.results = dict(current_state)

    try:
        logger.info("Starting pipeline execution for topic: %s", topic_val)
        final_state = run_research_pipeline(
            topic=topic_val,
            on_step_start=on_step_start,
            on_step_complete=on_step_complete,
            max_sources=5,
            max_refinements=2,
        )
        status_placeholder.empty()
        st.session_state.results = dict(final_state)
        st.session_state.running = False
        st.session_state.done = True
        st.rerun()
    except Exception as exc:
        logger.exception("Pipeline execution failed: %s", exc)
        status_placeholder.empty()
        st.session_state.running = False
        st.session_state.done = False
        sanitized_msg = sanitize_error_message(str(exc))
        st.error(f"The research pipeline encountered an error: {sanitized_msg}")


# ── Results display ───────────────────────────────────────────────────────────
r = st.session_state.results

if r:
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="section-heading">Research Findings & Artifacts</div>', unsafe_allow_html=True)

    # 1. Multi-source Display
    sources_data = r.get("sources_data", [])
    if sources_data:
        scraped_count = sum(1 for s in sources_data if s.get("is_scraped"))
        with st.expander(f"📚 Collected Multi-Source References ({len(sources_data)} sources · {scraped_count} scraped)", expanded=False):
            for i, src in enumerate(sources_data, 1):
                is_scraped = src.get("is_scraped", False)
                badge_class = "scraped" if is_scraped else "fallback"
                badge_text = "Full Content Scraped" if is_scraped else f"Snippet Only ({src.get('error') or 'Inaccessible'})"
                snippet_text = src.get("content", "")[:350] if is_scraped else src.get("snippet", "")

                st.markdown(f"""
                <div class="source-item-card">
                    <span class="source-badge {badge_class}">{badge_text}</span>
                    <span class="source-title">[{i}] {src.get('title', 'Untitled')}</span>
                    <br>
                    <a href="{src.get('url', '#')}" target="_blank" class="source-url">🔗 {src.get('url', '')}</a>
                    <div class="source-snippet">{snippet_text}...</div>
                </div>
                """, unsafe_allow_html=True)

    # 2. Final Report Display
    if "writer" in r and r["writer"]:
        st.markdown("""
        <div class="report-panel">
            <div class="panel-label orange">📝 Final Research Report (6 Verified Sections)</div>
        """, unsafe_allow_html=True)
        st.markdown(r["writer"])
        st.markdown("</div>", unsafe_allow_html=True)

        # Markdown Report Download
        st.download_button(
            label="⬇  Download Research Report (.md)",
            data=r["writer"],
            file_name=f"researchmind_report_{int(time.time())}.md",
            mime="text/markdown",
        )

    # 3. Critic Evaluation & Refinement Status
    if "critic" in r and r["critic"]:
        evaluations = r.get("evaluations", [])
        refinements = r.get("refinement_attempts", 0)

        latest_eval = evaluations[-1] if evaluations else {}
        score = latest_eval.get("score", 8.0)

        st.markdown(f"""
        <div class="feedback-panel">
            <div class="panel-label green">🧐 Critic Audit & Refinement Verification</div>
            <div class="score-banner">
                <div>
                    <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#80c890;text-transform:uppercase;">Quality Score</span><br>
                    <span class="score-pill">{score:.1f} / 10</span>
                </div>
                <div style="border-left:1px solid rgba(255,255,255,0.1);padding-left:1.5rem;">
                    <span style="font-family:'DM Mono',monospace;font-size:0.7rem;color:#a09890;text-transform:uppercase;">Refinement Status</span><br>
                    <span style="font-family:'DM Sans',sans-serif;font-size:0.95rem;color:#f0ebe0;">
                        {f'Completed {refinements} refinement cycle(s)' if refinements > 0 else 'Approved on initial evaluation'}
                    </span>
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown(r["critic"])
        st.markdown("</div>", unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="notice">
    ResearchMind · Multi-Agent Research System · Groq + LangChain + Tavily · Built with Streamlit
</div>
""", unsafe_allow_html=True)
