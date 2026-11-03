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

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchMind · AI-Powered Multi-Agent Research System",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Modern UI Stylesheet ──────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

/* ── Typography & Reset ── */
html, body, [class*="css"], [data-testid="stAppViewContainer"] {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
}

/* ── App Background ── */
.stApp {
    background: #090d16;
    background-image: 
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(99, 102, 241, 0.15) 0%, transparent 70%),
        radial-gradient(ellipse 60% 40% at 85% 105%, rgba(249, 115, 22, 0.08) 0%, transparent 60%);
    color: #f1f5f9;
}

/* ── Main Container Padding ── */
.block-container {
    padding-top: 2rem;
    padding-bottom: 4rem;
    max-width: 1200px;
}

/* ── Hero Section ── */
.hero-wrapper {
    text-align: center;
    padding: 2.2rem 1rem 1.8rem;
    margin-bottom: 1.5rem;
}
.hero-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.3);
    color: #a5b4fc;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.4rem 1rem;
    border-radius: 9999px;
    margin-bottom: 1rem;
}
.hero-title {
    font-size: clamp(2.5rem, 5.5vw, 4.2rem);
    font-weight: 800;
    line-height: 1.1;
    letter-spacing: -0.03em;
    color: #ffffff;
    margin-bottom: 0.8rem;
}
.hero-title-gradient {
    background: linear-gradient(135deg, #f97316 0%, #fb923c 45%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-subtitle {
    font-size: 1.05rem;
    color: #94a3b8;
    max-width: 650px;
    margin: 0 auto;
    line-height: 1.6;
    font-weight: 400;
}

/* ── Status Pills & Badges ── */
.badge-ready {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(16, 185, 129, 0.1);
    border: 1px solid rgba(16, 185, 129, 0.25);
    color: #34d399;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.75rem;
    border-radius: 6px;
}
.badge-warning {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.3);
    color: #fbbf24;
    font-size: 0.75rem;
    font-weight: 600;
    padding: 0.25rem 0.75rem;
    border-radius: 6px;
}

/* ── Step Cards ── */
.step-card {
    background: rgba(17, 24, 39, 0.7);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.1rem 1.3rem;
    margin-bottom: 0.75rem;
    transition: all 0.25s ease;
}
.step-card.active {
    background: rgba(99, 102, 241, 0.12);
    border-color: rgba(99, 102, 241, 0.55);
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.2);
}
.step-card.done {
    background: rgba(16, 185, 129, 0.08);
    border-color: rgba(16, 185, 129, 0.4);
}
.step-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.step-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    font-weight: 700;
    color: #f97316;
}
.step-title {
    font-weight: 600;
    font-size: 0.95rem;
    color: #f8fafc;
}
.step-status {
    margin-left: auto;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.05em;
    font-weight: 600;
}
.status-waiting { color: #64748b; }
.status-running { color: #818cf8; font-weight: 700; }
.status-done    { color: #34d399; font-weight: 700; }
.step-desc {
    font-size: 0.8rem;
    color: #94a3b8;
    margin-top: 0.3rem;
}

/* ── Source Cards ── */
.source-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-bottom: 0.85rem;
}
.source-tag {
    display: inline-block;
    font-size: 0.68rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    margin-bottom: 0.35rem;
}
.source-tag.scraped {
    background: rgba(16, 185, 129, 0.15);
    color: #34d399;
    border: 1px solid rgba(16, 185, 129, 0.3);
}
.source-tag.fallback {
    background: rgba(245, 158, 11, 0.15);
    color: #fbbf24;
    border: 1px solid rgba(245, 158, 11, 0.3);
}
.source-link {
    color: #818cf8 !important;
    font-weight: 600;
    font-size: 0.95rem;
    text-decoration: none;
}
.source-link:hover {
    text-decoration: underline;
}
.source-snippet {
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 0.4rem;
    line-height: 1.5;
}

/* ── Score Banner ── */
.score-box {
    display: flex;
    align-items: center;
    gap: 1.5rem;
    background: rgba(15, 23, 42, 0.8);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 12px;
    padding: 1.2rem 1.8rem;
    margin-bottom: 1.5rem;
}
.score-number {
    font-size: 2.2rem;
    font-weight: 800;
    color: #34d399;
    line-height: 1;
}

/* ── Custom Button ── */
.stButton > button {
    background: linear-gradient(135deg, #f97316 0%, #ea580c 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.75rem 2rem !important;
    box-shadow: 0 4px 18px rgba(249, 115, 22, 0.3) !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 24px rgba(249, 115, 22, 0.45) !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
for key in ("results", "running", "done", "topic_input"):
    if key not in st.session_state:
        st.session_state[key] = {} if key == "results" else ("" if key == "topic_input" else False)

# ── Hero Section ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-badge">✦ Multi-Agent AI System · Powered by Groq & Tavily</div>
    <h1 class="hero-title">Research<span class="hero-title-gradient">Mind</span></h1>
    <p class="hero-subtitle">
        Four specialized AI agents collaborate — exploring diverse web sources, deep-reading primary references,
        synthesizing structured reports, and running iterative critic audits.
    </p>
</div>
""", unsafe_allow_html=True)

# ── System Credentials Check ──────────────────────────────────────────────────
groq_key = get_credential("GROQ_API_KEY")
tavily_key = get_credential("TAVILY_API_KEY")
groq_model = get_credential("GROQ_MODEL", "openai/gpt-oss-120b")

groq_ready = bool(groq_key)
tavily_ready = bool(tavily_key)
system_ready = groq_ready and tavily_ready

# Status bar
stat_col1, stat_col2 = st.columns([3, 2])
with stat_col1:
    st.markdown(f"""
    <div style="font-size:0.85rem;color:#94a3b8;padding:0.4rem 0;">
        ⚡ <b>Engine:</b> Groq LPU &bull; <b>Model:</b> <code>{groq_model}</code>
    </div>
    """, unsafe_allow_html=True)

with stat_col2:
    if system_ready:
        st.markdown("""
        <div style="text-align:right;">
            <span class="badge-ready">✓ System Ready · API Keys Connected</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        missing = []
        if not groq_ready:
            missing.append("GROQ_API_KEY")
        if not tavily_ready:
            missing.append("TAVILY_API_KEY")
        st.markdown(f"""
        <div style="text-align:right;">
            <span class="badge-warning">⚠️ Missing {', '.join(missing)}</span>
        </div>
        """, unsafe_allow_html=True)

# ── Cloud Secrets Guidance Banner (if keys are missing) ─────────────────────────
if not system_ready:
    with st.expander("🔑 How to configure your API keys (Streamlit Cloud & Local)", expanded=True):
        st.markdown("""
        ### Setup Instructions:
        
        #### 1. On Streamlit Community Cloud:
        1. Click the **three dots `⋮`** at the bottom-right of your screen.
        2. Select **Settings** ⚙️ from the menu.
        3. Click on the **Secrets** tab in the popup.
        4. Paste your API keys in the text box:
        ```toml
        GROQ_API_KEY = "your_actual_groq_api_key_here"
        TAVILY_API_KEY = "your_actual_tavily_api_key_here"
        GROQ_MODEL = "openai/gpt-oss-120b"
        ```
        5. Click **Save**! The app will instantly detect the keys and reload.
        
        #### 2. On Local Machine:
        Add `GROQ_API_KEY` and `TAVILY_API_KEY` to your local `.env` file.
        """)

# ── Main Layout: Input & Pipeline ─────────────────────────────────────────────
st.write("")
col_left, col_space, col_right = st.columns([5, 0.4, 4.6])

with col_left:
    st.markdown("#### Research Topic")
    
    with st.container(border=True):
        topic_value = st.text_input(
            "Enter research subject or inquiry",
            value=st.session_state.topic_input,
            placeholder="e.g. CRISPR gene editing clinical breakthroughs in 2025",
            label_visibility="collapsed",
            key="user_topic_field"
        )
        run_btn = st.button("🚀  Run Research Pipeline", use_container_width=True)

    # Interactive Example Chips
    st.caption("💡 Quick-fill example topics:")
    ex_cols = st.columns(3)
    if ex_cols[0].button("🧬 CRISPR 2025", use_container_width=True, help="Load CRISPR gene editing topic"):
        st.session_state.topic_input = "CRISPR gene editing clinical breakthroughs in 2025"
        st.rerun()
    if ex_cols[1].button("⚛️ Quantum Computing", use_container_width=True, help="Load Quantum computing topic"):
        st.session_state.topic_input = "Quantum computing error mitigation architectures"
        st.rerun()
    if ex_cols[2].button("🤖 AI Agents", use_container_width=True, help="Load Autonomous AI Agents topic"):
        st.session_state.topic_input = "Autonomous AI agents in software engineering"
        st.rerun()

with col_right:
    st.markdown("#### Agent Pipeline Workflow")
    cards_placeholder = st.empty()

    def generate_cards_html(active_step: str = "", current_res: dict = None) -> str:
        res = current_res if current_res is not None else st.session_state.results
        steps = [
            ("01", "Search Agent", "search", "Multi-source web discovery & URL deduplication"),
            ("02", "Reader Agent", "reader", "Deep-content extraction from 3–5 unique URLs with fallback"),
            ("03", "Writer Agent", "writer", "Synthesizes structured 6-section academic report"),
            ("04", "Critic Agent", "critic", "Audits evidence, scores depth, and triggers refinement (max 2 loops)"),
        ]

        html_parts = []
        for num, title, key, desc in steps:
            if key in res and res[key]:
                state_label = "✓ DONE"
                state_cls = "status-done"
                card_cls = "done"
            elif active_step == key or (active_step == "refiner" and key == "writer"):
                state_label = "● ACTIVE"
                state_cls = "status-running"
                card_cls = "active"
            else:
                state_label = "WAITING"
                state_cls = "status-waiting"
                card_cls = ""

            html_parts.append(f"""
            <div class="step-card {card_cls}">
                <div class="step-header">
                    <span class="step-num">{num}</span>
                    <span class="step-title">{title}</span>
                    <span class="step-status {state_cls}">{state_label}</span>
                </div>
                <div class="step-desc">{desc}</div>
            </div>
            """)
        return "".join(html_parts)

    cards_placeholder.markdown(generate_cards_html(), unsafe_allow_html=True)

# ── Pipeline Execution Controller ─────────────────────────────────────────────
if run_btn:
    chosen_topic = topic_value.strip() if topic_value else st.session_state.topic_input.strip()
    
    try:
        clean_topic = validate_topic(chosen_topic)
    except ValueError as ve:
        st.warning(f"⚠️ {str(ve)}")
        clean_topic = None

    if clean_topic:
        if not system_ready:
            st.error("Cannot start pipeline: missing required API keys. Please add `GROQ_API_KEY` and `TAVILY_API_KEY` in Streamlit Secrets or `.env`.")
        else:
            st.session_state.topic_input = clean_topic
            st.session_state.results = {}
            st.session_state.running = True
            st.session_state.done = False
            st.rerun()

if st.session_state.running and not st.session_state.done:
    active_topic = st.session_state.topic_input
    status_bar = st.empty()

    def on_step_start(step_key: str, step_title: str, description: str):
        icons = {"search": "🔍", "reader": "📄", "writer": "✍️", "critic": "🧐", "refiner": "🔄"}
        icon = icons.get(step_key, "⚡")
        status_bar.info(f"{icon} **{step_title}**: {description}")
        cards_placeholder.markdown(
            generate_cards_html(active_step=step_key, current_res=st.session_state.results),
            unsafe_allow_html=True
        )

    def on_step_complete(step_key: str, output: str, current_state: dict):
        st.session_state.results = dict(current_state)
        cards_placeholder.markdown(
            generate_cards_html(active_step="", current_res=current_state),
            unsafe_allow_html=True
        )

    try:
        logger.info("Executing pipeline for topic: %s", active_topic)
        final_state = run_research_pipeline(
            topic=active_topic,
            on_step_start=on_step_start,
            on_step_complete=on_step_complete,
            max_sources=5,
            max_refinements=2,
        )
        status_bar.empty()
        st.session_state.results = dict(final_state)
        st.session_state.running = False
        st.session_state.done = True
        st.rerun()
    except Exception as exc:
        logger.exception("Pipeline execution error: %s", exc)
        status_bar.empty()
        st.session_state.running = False
        st.session_state.done = False
        clean_err = sanitize_error_message(str(exc))
        st.error(f"Pipeline execution encountered an error: {clean_err}")

# ── Results Presentation ──────────────────────────────────────────────────────
r = st.session_state.results

if r and ("writer" in r or "report" in r):
    st.divider()
    st.markdown("### 📊 Research Artifacts & Intelligence")

    tab_report, tab_sources, tab_critic = st.tabs([
        "📑  Final Research Report",
        "🌐  Collected Sources & Evidence",
        "🧐  Critic Audit & Refinement History",
    ])

    # ── Tab 1: Final Report ──
    with tab_report:
        report_markdown = r.get("writer", "") or r.get("report", "")
        
        col_btn1, col_btn2 = st.columns([4, 1])
        with col_btn2:
            st.download_button(
                label="⬇  Export Report (.md)",
                data=report_markdown,
                file_name=f"researchmind_report_{int(time.time())}.md",
                mime="text/markdown",
                use_container_width=True,
            )

        with st.container(border=True):
            st.markdown(report_markdown)

    # ── Tab 2: Sources & Evidence ──
    with tab_sources:
        sources_list = r.get("sources_data", [])
        if sources_list:
            scraped_count = sum(1 for s in sources_list if s.get("is_scraped"))
            st.caption(f"Gathered {len(sources_list)} primary references ({scraped_count} full pages scraped, {len(sources_list) - scraped_count} snippet fallbacks):")
            
            for idx, s in enumerate(sources_list, 1):
                is_scraped = s.get("is_scraped", False)
                tag_cls = "scraped" if is_scraped else "fallback"
                tag_label = "Full Content Scraped" if is_scraped else f"Search Snippet Only ({s.get('error') or 'Inaccessible'})"
                content_preview = s.get("content", "")[:350] if is_scraped else s.get("snippet", "")

                st.markdown(f"""
                <div class="source-card">
                    <span class="source-tag {tag_cls}">{tag_label}</span>
                    <br>
                    <a href="{s.get('url', '#')}" target="_blank" class="source-link">[{idx}] {s.get('title', 'Untitled')} ↗</a>
                    <div class="source-snippet">{content_preview}...</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No structured sources available.")

    # ── Tab 3: Critic Audit & Refinements ──
    with tab_critic:
        evaluations = r.get("evaluations", [])
        refinements = r.get("refinement_attempts", 0)
        latest_eval = evaluations[-1] if evaluations else {}
        score = latest_eval.get("score", 8.0)
        verdict = latest_eval.get("verdict", "Evaluation completed.")

        st.markdown(f"""
        <div class="score-box">
            <div>
                <div style="font-size:0.75rem;font-weight:700;color:#94a3b8;text-transform:uppercase;">Quality Score</div>
                <div class="score-number">{score:.1f} <span style="font-size:1.1rem;color:#94a3b8;">/ 10</span></div>
            </div>
            <div style="border-left:1px solid rgba(255,255,255,0.1);padding-left:1.5rem;">
                <div style="font-size:0.75rem;font-weight:700;color:#94a3b8;text-transform:uppercase;">Refinement History</div>
                <div style="font-size:1rem;color:#f8fafc;font-weight:600;">
                    {f'Completed {refinements} revision cycle(s)' if refinements > 0 else 'Approved on initial evaluation (Zero revisions required)'}
                </div>
                <div style="font-size:0.85rem;color:#94a3b8;margin-top:0.2rem;">
                    Verdict: <em>{verdict}</em>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        critic_markdown = r.get("critic", "")
        with st.container(border=True):
            st.markdown(critic_markdown)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;color:#64748b;font-size:0.8rem;margin-top:4rem;padding-top:1.5rem;border-top:1px solid rgba(255,255,255,0.06);">
    ResearchMind &bull; Autonomous Multi-Agent Research System &bull; Powered by Groq, LangChain & Tavily
</div>
""", unsafe_allow_html=True)
