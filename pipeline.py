import os
import logging
from typing import Callable, Dict, Optional, Any, List
from dotenv import load_dotenv

from agents import (
    execute_search_agent,
    execute_reader_agent,
    generate_report,
    evaluate_report,
    build_search_agent,
    build_reader_agent,
    get_writer_chain,
    get_critic_chain,
)
from models import SourceItem, ResearchReport, CriticEvaluation, PipelineResult
from tools import sanitize_error_message, get_credential

load_dotenv()

logger = logging.getLogger("researchmind.pipeline")

MAX_REFINEMENT_ATTEMPTS = 2
MIN_ACCEPTABLE_SCORE = 8.0


def validate_topic(topic: str) -> str:
    """Validate research topic input."""
    if not topic or not isinstance(topic, str):
        raise ValueError("Research topic must be a non-empty string.")
    cleaned = topic.strip()
    if len(cleaned) < 3:
        raise ValueError("Research topic is too short. Please provide at least 3 characters.")
    if len(cleaned) > 500:
        raise ValueError("Research topic is too long (maximum 500 characters).")
    return cleaned


def validate_environment() -> None:
    """Validate that required environment variables or Streamlit secrets are set."""
    missing_keys = []
    if not get_credential("GROQ_API_KEY"):
        missing_keys.append("GROQ_API_KEY")
    if not get_credential("TAVILY_API_KEY"):
        missing_keys.append("TAVILY_API_KEY")

    if missing_keys:
        keys_str = ", ".join(missing_keys)
        raise ValueError(f"Missing required API key(s): {keys_str}. Please configure them in your .env file or Streamlit secrets.")


def run_research_pipeline(
    topic: str,
    on_step_start: Optional[Callable[[str, str, str], None]] = None,
    on_step_complete: Optional[Callable[[str, str, Dict[str, Any]], None]] = None,
    max_sources: int = 5,
    max_refinements: int = MAX_REFINEMENT_ATTEMPTS,
) -> Dict[str, Any]:
    """
    Execute the full ResearchMind multi-agent research pipeline.
    
    Sequential Agent Workflow:
    Research Topic
        ↓
    Search Agent (multi-source discovery & deduplication)
        ↓
    Reader Agent (extracts content from 3-5 URLs, handles failures gracefully)
        ↓
    Writer Agent (synthesizes structured 6-section report with verified URLs)
        ↓
    Critic Agent (evaluates factual grounding, score & missing info)
        ↓ (up to 2 refinement attempts if improvements needed)
    Final Report
    """
    clean_topic = validate_topic(topic)
    validate_environment()

    state: Dict[str, Any] = {
        "topic": clean_topic,
        "sources": [],
        "sources_data": [],
        "search": "",
        "reader": "",
        "writer": "",
        "critic": "",
        "search_results": "",
        "scraped_content": "",
        "report": "",
        "feedback": "",
        "refinement_attempts": 0,
        "evaluations": [],
    }

    # ── Step 1: Search Agent ──────────────────────────────────────────────────
    step1_title = "Search Agent"
    step1_desc = "Discovering authoritative, multi-source web references..."
    logger.info("=" * 60)
    logger.info("Step 1: %s - %s", step1_title, step1_desc)
    logger.info("=" * 60)
    if on_step_start:
        on_step_start("search", step1_title, step1_desc)

    try:
        raw_sources = execute_search_agent(clean_topic, max_results=max_sources + 2)
        if not raw_sources:
            logger.warning("No search results returned for '%s'. Retrying with broader keywords.", clean_topic)
            raw_sources = execute_search_agent(clean_topic.split(":")[0], max_results=max_sources)
    except Exception as e:
        logger.error("Search Agent failed: %s", e)
        raise RuntimeError(sanitize_error_message(f"Search Agent failed: {str(e)}")) from e

    # Format search preview
    search_preview_lines = []
    for s in raw_sources:
        search_preview_lines.append(f"Title: {s.get('title')}\nURL: {s.get('url')}\nSnippet: {s.get('snippet')[:250]}\n")
    search_text = "\n----\n".join(search_preview_lines) if search_preview_lines else "No search results found."
    state["search"] = search_text
    state["search_results"] = search_text
    logger.info("Step 1 complete. Found %d unique sources.", len(raw_sources))
    if on_step_complete:
        on_step_complete("search", search_text, dict(state))

    # ── Step 2: Reader Agent ──────────────────────────────────────────────────
    step2_title = "Reader Agent"
    step2_desc = f"Extracting clean content from top {min(len(raw_sources), max_sources)} unique sources..."
    logger.info("=" * 60)
    logger.info("Step 2: %s - %s", step2_title, step2_desc)
    logger.info("=" * 60)
    if on_step_start:
        on_step_start("reader", step2_title, step2_desc)

    try:
        source_items = execute_reader_agent(raw_sources, max_urls=max_sources)
    except Exception as e:
        logger.error("Reader Agent failed: %s", e)
        raise RuntimeError(sanitize_error_message(f"Reader Agent failed: {str(e)}")) from e

    state["sources"] = source_items
    state["sources_data"] = [s.model_dump() for s in source_items]

    # Format reader preview
    reader_preview_lines = []
    for s in source_items:
        status_tag = "Scraped Successfully" if s.is_scraped else f"Snippet Fallback ({s.error or 'Inaccessible'})"
        reader_preview_lines.append(
            f"Source: {s.title}\nURL: {s.url}\nStatus: {status_tag}\nContent Preview:\n{s.to_display_snippet(300)}\n"
        )
    reader_text = "\n----\n".join(reader_preview_lines)
    state["reader"] = reader_text
    state["scraped_content"] = reader_text
    logger.info("Step 2 complete. Gathered %d sources (%d scraped, %d snippet fallbacks).",
                len(source_items),
                sum(1 for s in source_items if s.is_scraped),
                sum(1 for s in source_items if not s.is_scraped))
    if on_step_complete:
        on_step_complete("reader", reader_text, dict(state))

    # ── Step 3: Writer Agent (Initial Draft) ──────────────────────────────────
    step3_title = "Writer Agent"
    step3_desc = "Drafting comprehensive research report with 6 verified sections..."
    logger.info("=" * 60)
    logger.info("Step 3: %s - %s", step3_title, step3_desc)
    logger.info("=" * 60)
    if on_step_start:
        on_step_start("writer", step3_title, step3_desc)

    try:
        current_report = generate_report(topic=clean_topic, sources=source_items)
    except Exception as e:
        logger.error("Writer Agent failed: %s", e)
        raise RuntimeError(sanitize_error_message(f"Writer Agent failed: {str(e)}")) from e

    state["writer"] = current_report.to_markdown()
    state["report"] = current_report.to_markdown()
    state["report_model"] = current_report
    logger.info("Step 3 complete. Initial report drafted (%d chars).", len(current_report.to_markdown()))
    if on_step_complete:
        on_step_complete("writer", current_report.to_markdown(), dict(state))

    # ── Step 4: Critic Agent & Refinement Loop ────────────────────────────────
    evaluations: List[CriticEvaluation] = []
    refinement_count = 0

    while True:
        eval_num = refinement_count + 1
        step4_title = f"Critic Agent (Evaluation {eval_num})"
        step4_desc = "Auditing report for factual grounding, completeness, and structure..."
        logger.info("=" * 60)
        logger.info("Step 4: %s - %s", step4_title, step4_desc)
        logger.info("=" * 60)
        if on_step_start:
            on_step_start("critic", step4_title, step4_desc)

        try:
            evaluation = evaluate_report(
                topic=clean_topic,
                report=current_report,
                sources=source_items,
            )
        except Exception as e:
            logger.error("Critic Agent failed: %s", e)
            raise RuntimeError(sanitize_error_message(f"Critic Agent failed: {str(e)}")) from e

        evaluations.append(evaluation)
        state["critic"] = evaluation.to_markdown()
        state["feedback"] = evaluation.to_markdown()
        state["critic_model"] = evaluation
        state["evaluations"] = [ev.model_dump() for ev in evaluations]
        state["refinement_attempts"] = refinement_count

        logger.info("Critic Evaluation %d: Score = %.1f/10, Needs Refinement = %s",
                    eval_num, evaluation.score, evaluation.needs_refinement)
        if on_step_complete:
            on_step_complete("critic", evaluation.to_markdown(), dict(state))

        # Check termination condition
        if not evaluation.needs_refinement or evaluation.score >= MIN_ACCEPTABLE_SCORE:
            logger.info("Report approved by Critic with score %.1f/10.", evaluation.score)
            break

        if refinement_count >= max_refinements:
            logger.info("Reached maximum refinement limit (%d attempts). Concluding pipeline.", max_refinements)
            break

        # Begin refinement iteration
        refinement_count += 1
        refine_title = f"Writer Agent (Refinement {refinement_count}/{max_refinements})"
        refine_desc = (
            f"Refining report based on Critic feedback (Score: {evaluation.score:.1f}/10)..."
        )
        logger.info("=" * 60)
        logger.info("Refinement Loop: %s - %s", refine_title, refine_desc)
        logger.info("=" * 60)
        if on_step_start:
            on_step_start("writer", refine_title, refine_desc)

        feedback_summary = (
            f"Score: {evaluation.score:.1f}/10\n"
            f"Missing Info: {', '.join(evaluation.missing_information) if evaluation.missing_information else 'None'}\n"
            f"Weak Explanations: {', '.join(evaluation.weak_explanations) if evaluation.weak_explanations else 'None'}\n"
            f"Unsupported Claims: {', '.join(evaluation.unsupported_claims) if evaluation.unsupported_claims else 'None'}\n"
            f"Recommendations: {', '.join(evaluation.actionable_recommendations) if evaluation.actionable_recommendations else 'Improve overall depth'}\n"
        )

        try:
            current_report = generate_report(
                topic=clean_topic,
                sources=source_items,
                refinement_feedback=feedback_summary,
                previous_report=current_report,
            )
        except Exception as e:
            logger.error("Writer refinement failed: %s. Keeping current draft.", e)
            break

        state["writer"] = current_report.to_markdown()
        state["report"] = current_report.to_markdown()
        state["report_model"] = current_report
        state["refinement_attempts"] = refinement_count
        if on_step_complete:
            on_step_complete("writer", current_report.to_markdown(), dict(state))

    # Construct final pipeline result
    pipeline_result = PipelineResult(
        topic=clean_topic,
        sources=source_items,
        report=current_report,
        critic_evaluations=evaluations,
        final_critic_evaluation=evaluations[-1] if evaluations else None,
        refinement_attempts=refinement_count,
        raw_markdown_report=current_report.to_markdown(),
    )

    final_state = pipeline_result.to_state_dict()
    logger.info("Pipeline execution successfully completed for '%s' with %d refinement(s).",
                clean_topic, refinement_count)
    return final_state


if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        user_topic = input("\nEnter a research topic: ")
        result = run_research_pipeline(user_topic)
        print("\n" + "=" * 60)
        print("FINAL RESEARCH REPORT:")
        print("=" * 60)
        print(result.get("report", ""))
        print("\n" + "=" * 60)
        print("FINAL CRITIC EVALUATION:")
        print("=" * 60)
        print(result.get("critic", ""))
    except Exception as exc:
        logger.error("Pipeline run failed: %s", exc)
        sys.exit(1)
