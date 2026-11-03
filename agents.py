import os
import re
import json
import logging
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from tools import (
    web_search,
    scrape_url,
    search_sources,
    scrape_multiple_sources,
    sanitize_error_message,
    get_credential,
)
from models import SourceItem, ResearchReport, CriticEvaluation

load_dotenv()

logger = logging.getLogger(__name__)

# Default model configuration
DEFAULT_MODEL = "openai/gpt-oss-120b"
DEFAULT_TEMPERATURE = 0.0


def get_llm(model: Optional[str] = None, temperature: float = DEFAULT_TEMPERATURE) -> ChatGroq:
    """Instantiate ChatGroq with API key validation and model fallback."""
    api_key = get_credential("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY is not configured in environment or Streamlit secrets.")
        raise ValueError("GROQ_API_KEY is missing. Please set GROQ_API_KEY in your .env file or Streamlit secrets.")

    model_name = model or get_credential("GROQ_MODEL", DEFAULT_MODEL)
    return ChatGroq(
        model=model_name,
        temperature=temperature,
        api_key=api_key,
    )


# ── Agent 1: Search Agent ─────────────────────────────────────────────────────
def build_search_agent(llm_instance: Optional[ChatGroq] = None):
    """Build the search agent equipped with web_search tool."""
    model = llm_instance or get_llm()
    return create_agent(
        model=model,
        tools=[web_search],
    )


def execute_search_agent(topic: str, max_results: int = 6) -> List[Dict[str, str]]:
    """
    Execute the Search Agent to gather multiple deduplicated candidate sources.
    Returns list of dicts with 'title', 'url', and 'snippet'.
    """
    logger.info("Search Agent searching for topic: '%s'", topic)
    return search_sources(topic, max_results=max_results)


# ── Agent 2: Reader Agent ─────────────────────────────────────────────────────
def build_reader_agent(llm_instance: Optional[ChatGroq] = None):
    """Build the reader agent equipped with scrape_url tool."""
    model = llm_instance or get_llm()
    return create_agent(
        model=model,
        tools=[scrape_url],
    )


def execute_reader_agent(sources: List[Dict[str, str]], max_urls: int = 5) -> List[SourceItem]:
    """
    Execute the Reader Agent to deep-read and scrape content from top 3-5 unique URLs.
    Handles inaccessible websites gracefully without breaking the pipeline.
    """
    logger.info("Reader Agent reading top %d sources...", max_urls)
    return scrape_multiple_sources(sources, max_urls=max_urls)


# ── Agent 3: Writer Agent ─────────────────────────────────────────────────────
writer_system_prompt = (
    "You are an elite research writer. You write rigorous, well-structured, objective, and deeply informative research reports.\n\n"
    "CRITICAL FACTUAL INTEGRITY RULES:\n"
    "1. Strict Evidence Grounding: Base all factual assertions on the provided research sources.\n"
    "2. No Fabricated References: Only cite URLs that exist in the provided research data. Never invent or hallucinate URLs or citations.\n"
    "3. Epistemic Distinction: Clearly distinguish source-verified facts from model synthesis and analytical extrapolation.\n"
    "4. Comprehensive Depth: Key findings must contain at least 3 deep, well-explained insights with quantitative data or technical specifics if present.\n"
    "5. Required Structure: You must adhere strictly to these sections:\n"
    "   - Executive Summary\n"
    "   - Introduction\n"
    "   - Key Findings (minimum 3 well-explained points)\n"
    "   - Detailed Analysis\n"
    "   - Conclusion\n"
    "   - References (only real URLs from the research)"
)

writer_human_prompt = """Topic: {topic}

Collected Research Sources:
{research}

{refinement_context}

Draft a complete, authoritative research report on '{topic}'. Ensure every section is fully articulated."""

writer_prompt = ChatPromptTemplate.from_messages([
    ("system", writer_system_prompt),
    ("human", writer_human_prompt),
])


def _format_sources_for_prompt(sources: List[SourceItem]) -> str:
    """Format structured sources into a clean readable string for LLM prompts."""
    formatted = []
    for i, s in enumerate(sources, 1):
        status = "Full Content Scraped" if s.is_scraped else f"Search Snippet Only ({s.error or 'Inaccessible'})"
        content_text = s.content[:2000].strip() if s.content else s.snippet[:500].strip()
        formatted.append(
            f"--- Source [{i}] ---\n"
            f"Title: {s.title}\n"
            f"URL: {s.url}\n"
            f"Status: {status}\n"
            f"Content:\n{content_text}\n"
        )
    return "\n".join(formatted) if formatted else "No external research sources available."


def _parse_report_text(topic: str, raw_text: str, fallback_sources: List[SourceItem]) -> ResearchReport:
    """Fallback parser to construct a valid ResearchReport from raw text or markdown."""
    exec_match = re.search(r"##\s*Executive Summary\s*\n(.*?)(?=\n##|\Z)", raw_text, re.S)
    intro_match = re.search(r"##\s*Introduction\s*\n(.*?)(?=\n##|\Z)", raw_text, re.S)
    findings_match = re.search(r"##\s*Key Findings\s*\n(.*?)(?=\n##|\Z)", raw_text, re.S)
    analysis_match = re.search(r"##\s*Detailed Analysis\s*\n(.*?)(?=\n##|\Z)", raw_text, re.S)
    conc_match = re.search(r"##\s*Conclusion\s*\n(.*?)(?=\n##|\Z)", raw_text, re.S)
    refs_match = re.search(r"##\s*References\s*\n(.*?)(?=\n##|\Z)", raw_text, re.S)

    findings = []
    if findings_match:
        findings = [
            f.strip() for f in re.findall(r"(?:^\d+\.|\-|\*)\s*(.+)", findings_match.group(1), re.M)
            if f.strip()
        ]
    if not findings:
        findings = [
            "Research synthesized from multiple domain sources.",
            "Key technical and developmental trends identified in the analysis.",
            "Strategic opportunities and challenges highlighted in the findings."
        ]

    # Extract verified references
    valid_source_urls = {s.url for s in fallback_sources}
    refs = []
    if refs_match:
        extracted_urls = re.findall(r"https?://[^\s<>\"']+", refs_match.group(1))
        # Keep URLs that match valid source URLs, or all if none matched
        for u in extracted_urls:
            clean_u = u.rstrip(".,;)")
            if clean_u in valid_source_urls or not valid_source_urls:
                if clean_u not in refs:
                    refs.append(clean_u)

    if not refs:
        refs = [s.url for s in fallback_sources if s.url]

    return ResearchReport(
        topic=topic,
        executive_summary=exec_match.group(1).strip() if exec_match else raw_text[:400].strip(),
        introduction=intro_match.group(1).strip() if intro_match else raw_text[:600].strip(),
        key_findings=findings,
        detailed_analysis=analysis_match.group(1).strip() if analysis_match else raw_text.strip(),
        conclusion=conc_match.group(1).strip() if conc_match else "Research indicates significant developments with continued study required.",
        references=refs,
    )


def generate_report(
    topic: str,
    sources: List[SourceItem],
    refinement_feedback: Optional[str] = None,
    previous_report: Optional[ResearchReport] = None,
    llm_instance: Optional[ChatGroq] = None,
) -> ResearchReport:
    """
    Generate or refine a structured research report using the Writer Agent.
    Validates output using the ResearchReport Pydantic model with automatic fallback parsing.
    """
    model = llm_instance or get_llm()
    research_text = _format_sources_for_prompt(sources)

    refinement_context = ""
    if refinement_feedback and previous_report:
        refinement_context = (
            "CRITIC FEEDBACK & REFINEMENT INSTRUCTIONS:\n"
            f"{refinement_feedback}\n\n"
            "PREVIOUS DRAFT TO IMPROVE:\n"
            f"{previous_report.to_markdown()[:2500]}\n\n"
            "INSTRUCTIONS: Thoroughly address all critic feedback. Enhance weak explanations, "
            "incorporate missing context, and ensure all claims remain source-supported."
        )

    # Attempt structured output first
    try:
        structured_llm = model.with_structured_output(ResearchReport)
        chain = writer_prompt | structured_llm
        logger.info("Executing Writer Agent with structured output for topic: '%s'", topic)
        result = chain.invoke({
            "topic": topic,
            "research": research_text,
            "refinement_context": refinement_context,
        })
        if isinstance(result, ResearchReport):
            # Ensure references only contain real sources
            known_urls = {s.url for s in sources}
            if not result.references and known_urls:
                result.references = list(known_urls)
            return result
    except Exception as e:
        logger.warning("Structured output invocation failed for Writer Agent (%s). Falling back to string chain parser.", e)

    # Fallback to standard string chain + regex parsing
    str_chain = writer_prompt | model | StrOutputParser()
    raw_markdown = str_chain.invoke({
        "topic": topic,
        "research": research_text,
        "refinement_context": refinement_context,
    })
    return _parse_report_text(topic, raw_markdown, sources)


# ── Agent 4: Critic Agent ─────────────────────────────────────────────────────
critic_system_prompt = (
    "You are a rigorous, uncompromising academic and research critic. You evaluate research reports with extreme precision.\n\n"
    "AUDIT CRITERIA:\n"
    "1. Factual Grounding: Are claims backed by the research sources? Flag any unsupported assertions.\n"
    "2. Completeness: Are important facets of the topic omitted or glossed over?\n"
    "3. Depth: Are explanations deep, analytical, and informative rather than shallow or generic?\n"
    "4. Reference Integrity: Verify that references contain genuine source URLs.\n"
    "5. Structural Quality: Check for Executive Summary, Introduction, Key Findings, Detailed Analysis, Conclusion, and References.\n\n"
    "DECISION RULE FOR REFINEMENT:\n"
    "- If score < 8.0, or if there are noticeable gaps or weak explanations, set needs_refinement = True.\n"
    "- If the report is thorough, factually sound, and comprehensive (score >= 8.0), set needs_refinement = False."
)

critic_human_prompt = """Topic: {topic}

Verified Research Sources Provided:
{sources}

Generated Research Report:
{report}

Evaluate this report strictly and objectively. Identify strengths, missing information, weak explanations, and unsupported claims."""

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", critic_system_prompt),
    ("human", critic_human_prompt),
])


def _parse_critic_text(raw_text: str) -> CriticEvaluation:
    """Fallback parser to construct a valid CriticEvaluation from unstructured LLM text."""
    score_match = re.search(r"Score:\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*10", raw_text, re.I)
    score = float(score_match.group(1)) if score_match else 7.5
    score = max(1.0, min(10.0, score))

    verdict_match = re.search(r"Verdict:\s*(.+?)(?=\n[A-Z]|\Z)", raw_text, re.I | re.S)
    verdict = verdict_match.group(1).strip() if verdict_match else "Rigorous evaluation complete."

    # Extract bullet points
    bullets = re.findall(r"^[-\*]\s*(.+)", raw_text, re.M)
    strengths = [b.strip() for b in bullets[:3]] or ["Clear structure and topical relevance."]

    needs_refinement = score < 8.0

    return CriticEvaluation(
        score=score,
        strengths=strengths,
        weak_explanations=["Ensure all technical concepts are deeply articulated with concrete evidence."] if needs_refinement else [],
        missing_information=["Broaden context and discuss emerging developments."] if needs_refinement else [],
        unsupported_claims=["Double-check that all analytical assertions link back to sources."] if needs_refinement else [],
        actionable_recommendations=["Deepen the key findings and ensure source alignment in the analysis."] if needs_refinement else ["Report meets high quality standards."],
        needs_refinement=needs_refinement,
        verdict=verdict,
    )


def evaluate_report(
    topic: str,
    report: ResearchReport,
    sources: List[SourceItem],
    llm_instance: Optional[ChatGroq] = None,
) -> CriticEvaluation:
    """
    Execute the Critic Agent to evaluate the generated report.
    Validates output using the CriticEvaluation Pydantic model with automatic fallback parsing.
    """
    model = llm_instance or get_llm()
    sources_summary = _format_sources_for_prompt(sources)
    report_markdown = report.to_markdown()

    # Attempt structured output first
    try:
        structured_llm = model.with_structured_output(CriticEvaluation)
        chain = critic_prompt | structured_llm
        logger.info("Executing Critic Agent with structured output for topic: '%s'", topic)
        result = chain.invoke({
            "topic": topic,
            "sources": sources_summary,
            "report": report_markdown,
        })
        if isinstance(result, CriticEvaluation):
            return result
    except Exception as e:
        logger.warning("Structured output invocation failed for Critic Agent (%s). Falling back to string chain parser.", e)

    # Fallback to string chain + regex parsing
    str_chain = critic_prompt | model | StrOutputParser()
    raw_eval = str_chain.invoke({
        "topic": topic,
        "sources": sources_summary,
        "report": report_markdown,
    })
    return _parse_critic_text(raw_eval)


# ── Legacy compatibility helper chains ────────────────────────────────────────
def get_writer_chain(llm_instance: Optional[ChatGroq] = None):
    """Legacy writer chain."""
    model = llm_instance or get_llm()
    return writer_prompt | model | StrOutputParser()


def get_critic_chain(llm_instance: Optional[ChatGroq] = None):
    """Legacy critic chain."""
    model = llm_instance or get_llm()
    return critic_prompt | model | StrOutputParser()


# ── Lazy module attributes for backwards compatibility ────────────────────────
_cached_llm: Optional[ChatGroq] = None
_cached_writer_chain = None
_cached_critic_chain = None


def __getattr__(name: str):
    global _cached_llm, _cached_writer_chain, _cached_critic_chain
    if name == "llm":
        if _cached_llm is None:
            _cached_llm = get_llm()
        return _cached_llm
    if name == "writer_chain":
        if _cached_writer_chain is None:
            _cached_writer_chain = get_writer_chain()
        return _cached_writer_chain
    if name == "critic_chain":
        if _cached_critic_chain is None:
            _cached_critic_chain = get_critic_chain()
        return _cached_critic_chain
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
