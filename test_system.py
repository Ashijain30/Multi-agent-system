import os
import unittest
from unittest.mock import patch, MagicMock

import tools
import agents
import pipeline
from models import SourceItem, ResearchReport, CriticEvaluation, PipelineResult


class TestResearchMindSystem(unittest.TestCase):
    def setUp(self):
        self.sample_sources = [
            SourceItem(
                title="CRISPR Advances 2025",
                url="https://example.com/crispr",
                snippet="Clinical trials show breakthrough in in-vivo base editing.",
                content="Clinical trials in 2025 demonstrated breakthrough precision in in-vivo base editing without double-strand breaks.",
                is_scraped=True,
            ),
            SourceItem(
                title="Gene Editing Regulatory Standards",
                url="https://example.com/fda-crispr",
                snippet="Regulatory pathways for gene therapy treatments.",
                content="FDA and EMA established joint harmonized standards for off-target assessment in therapeutic gene editing.",
                is_scraped=True,
            ),
            SourceItem(
                title="Delivery Mechanisms",
                url="https://example.com/lnp-delivery",
                snippet="Lipid nanoparticle delivery for CRISPR cas systems.",
                content="Targeted lipid nanoparticles enabled organ-specific delivery beyond liver tissue.",
                is_scraped=False,
                error="HTTP 403 Forbidden",
            ),
        ]

    # ── Test 1: Empty inputs and Topic Validation ─────────────────────────────
    def test_topic_validation(self):
        # Empty string
        with self.assertRaises(ValueError):
            pipeline.validate_topic("")
        
        # Whitespace
        with self.assertRaises(ValueError):
            pipeline.validate_topic("     ")

        # Too short (< 3 chars)
        with self.assertRaises(ValueError):
            pipeline.validate_topic("AI")

        # None
        with self.assertRaises(ValueError):
            pipeline.validate_topic(None)

        # Valid topic
        clean = pipeline.validate_topic("  Quantum computing breakthroughs  ")
        self.assertEqual(clean, "Quantum computing breakthroughs")

    # ── Test 2: URL Normalization and Deduplication ───────────────────────────
    def test_url_normalization_and_dedup(self):
        u1 = "https://example.com/research/paper/"
        u2 = "https://EXAMPLE.COM/research/paper"
        u3 = "https://example.com/research/paper?utm_source=twitter&utm_medium=social"
        u4 = "https://example.com/research/paper#section2"

        norm1 = tools.normalize_url(u1)
        norm2 = tools.normalize_url(u2)
        norm3 = tools.normalize_url(u3)
        norm4 = tools.normalize_url(u4)

        self.assertEqual(norm1, norm2)
        self.assertEqual(norm1, norm3)
        self.assertEqual(norm1, norm4)

        raw_items = [
            {"title": "Paper A", "url": u1, "snippet": "Snippet A"},
            {"title": "Paper A Dup", "url": u2, "snippet": "Snippet A dup"},
            {"title": "Paper B", "url": "https://example.com/other-topic", "snippet": "Snippet B"},
        ]

        with patch("tools.get_tavily_client") as mock_tavily:
            mock_client = MagicMock()
            mock_client.search.return_value = {"results": raw_items}
            mock_tavily.return_value = mock_client

            deduped = tools.search_sources("test topic", max_results=5)
            self.assertEqual(len(deduped), 2)
            self.assertEqual(deduped[0]["url"], u1)
            self.assertEqual(deduped[1]["url"], "https://example.com/other-topic")

    # ── Test 3: Inaccessible URLs do not stop Reader Agent ─────────────────────
    def test_inaccessible_websites_fallback(self):
        sources = [
            {"title": "Valid Site", "url": "https://example.com/valid", "snippet": "Valid snippet"},
            {"title": "404 Error Site", "url": "https://example.com/missing", "snippet": "Fallback snippet 1"},
            {"title": "Timeout Site", "url": "https://example.com/timeout", "snippet": "Fallback snippet 2"},
            {"title": "Bad Scheme", "url": "ftp://bad-scheme.com", "snippet": "Fallback snippet 3"},
        ]

        def fake_fetch(url):
            if "valid" in url:
                return "Full content successfully scraped from the webpage.", None
            elif "missing" in url:
                return None, "HTTP 404 error."
            elif "timeout" in url:
                return None, "Request timed out."
            return None, "Invalid URL"

        with patch("tools.fetch_url_content", side_effect=fake_fetch):
            results = tools.scrape_multiple_sources(sources, max_urls=4)
            self.assertEqual(len(results), 4)

            # Valid site scraped
            self.assertTrue(results[0].is_scraped)
            self.assertIn("Full content", results[0].content)

            # Inaccessible site 1: fell back to snippet without crashing
            self.assertFalse(results[1].is_scraped)
            self.assertEqual(results[1].content, "Fallback snippet 1")
            self.assertEqual(results[1].error, "HTTP 404 error.")

            # Inaccessible site 2: fell back to snippet
            self.assertFalse(results[2].is_scraped)
            self.assertEqual(results[2].content, "Fallback snippet 2")

            # Bad scheme: caught without crashing
            self.assertFalse(results[3].is_scraped)

    # ── Test 4: Structured Output Models and 6-Section Report ─────────────────
    def test_structured_report_model(self):
        report = ResearchReport(
            topic="Autonomous AI Agents",
            executive_summary="AI agents are revolutionizing software development.",
            introduction="From rule-based systems to LLM-driven autonomous workflows.",
            key_findings=[
                "Multi-agent collaboration outperforms monolithic LLM prompts.",
                "Deterministic verification loops reduce hallucination rates by over 60%.",
                "Tool integration and sandboxing provide safe runtime execution.",
            ],
            detailed_analysis="Evidence from recent benchmarks demonstrates that structured multi-agent systems maintain state better.",
            conclusion="Autonomous agents will become the standard paradigm for software engineering.",
            references=["https://example.com/agent-benchmark", "https://example.com/agent-paper"],
        )

        md = report.to_markdown()
        self.assertIn("## Executive Summary", md)
        self.assertIn("## Introduction", md)
        self.assertIn("## Key Findings", md)
        self.assertIn("## Detailed Analysis", md)
        self.assertIn("## Conclusion", md)
        self.assertIn("## References", md)
        self.assertIn("<https://example.com/agent-benchmark>", md)
        self.assertEqual(len(report.key_findings), 3)

    # ── Test 5: Critic Evaluation Model ───────────────────────────────────────
    def test_critic_evaluation_model(self):
        evaluation = CriticEvaluation(
            score=8.5,
            strengths=["Thorough key findings", "Well-grounded analysis"],
            weak_explanations=[],
            missing_information=[],
            unsupported_claims=[],
            actionable_recommendations=["Consider expanding on latency overhead."],
            needs_refinement=False,
            verdict="High-quality report meeting publication criteria.",
        )

        md = evaluation.to_markdown()
        self.assertIn("8.5 / 10", md)
        self.assertIn("[Approved]", md)
        self.assertFalse(evaluation.needs_refinement)

    # ── Test 6: Refinement Loop with Maximum 2 Iterations ─────────────────────
    def test_refinement_loop_enforces_max_attempts(self):
        # Critic always requests refinement (score 6.5)
        mock_eval_need_refine = CriticEvaluation(
            score=6.5,
            strengths=["Good start"],
            weak_explanations=["Section 3 needs more evidence"],
            missing_information=["Missing economic considerations"],
            unsupported_claims=["Claim about cost reduction lacks source"],
            actionable_recommendations=["Provide benchmark numbers"],
            needs_refinement=True,
            verdict="Requires further refinement.",
        )

        dummy_report = ResearchReport(
            topic="Testing Refinement Loop",
            executive_summary="Summary draft.",
            introduction="Intro draft.",
            key_findings=["Finding 1", "Finding 2", "Finding 3"],
            detailed_analysis="Analysis draft.",
            conclusion="Conclusion draft.",
            references=["https://example.com/ref1"],
        )

        with patch("pipeline.execute_search_agent", return_value=[{"title": "T1", "url": "https://example.com/ref1", "snippet": "S1"}]), \
             patch("pipeline.execute_reader_agent", return_value=self.sample_sources), \
             patch("pipeline.generate_report", return_value=dummy_report) as mock_writer, \
             patch("pipeline.evaluate_report", return_value=mock_eval_need_refine) as mock_critic, \
             patch.dict(os.environ, {"GROQ_API_KEY": "dummy_gsk", "TAVILY_API_KEY": "dummy_tvly"}):

            result = pipeline.run_research_pipeline("Testing Refinement Loop", max_refinements=2)

            # Initial evaluation + 2 refinement evaluations = 3 total critic evaluations
            self.assertEqual(mock_critic.call_count, 3)
            # Initial draft + 2 refinement drafts = 3 total writer calls
            self.assertEqual(mock_writer.call_count, 3)
            # Refinement attempts capped at 2
            self.assertEqual(result["refinement_attempts"], 2)

    # ── Test 7: Refinement Stops Early When Critic Approves (Score >= 8.0) ────
    def test_refinement_early_approval(self):
        # Evaluation 1: needs refinement (score 7.0)
        eval1 = CriticEvaluation(
            score=7.0,
            strengths=["Clear premise"],
            weak_explanations=["Shallow analysis"],
            missing_information=["Missing source link"],
            unsupported_claims=[],
            actionable_recommendations=["Add technical citations"],
            needs_refinement=True,
            verdict="Needs minor revision.",
        )
        # Evaluation 2: approved (score 9.0)
        eval2 = CriticEvaluation(
            score=9.0,
            strengths=["Exceptional depth", "Flawless citations"],
            weak_explanations=[],
            missing_information=[],
            unsupported_claims=[],
            actionable_recommendations=[],
            needs_refinement=False,
            verdict="Outstanding report.",
        )

        dummy_report = ResearchReport(
            topic="Early Approval Topic",
            executive_summary="Summary.",
            introduction="Intro.",
            key_findings=["Point 1", "Point 2", "Point 3"],
            detailed_analysis="Analysis.",
            conclusion="Conclusion.",
            references=["https://example.com/src1"],
        )

        with patch("pipeline.execute_search_agent", return_value=[{"title": "T1", "url": "https://example.com/src1", "snippet": "S1"}]), \
             patch("pipeline.execute_reader_agent", return_value=self.sample_sources), \
             patch("pipeline.generate_report", return_value=dummy_report) as mock_writer, \
             patch("pipeline.evaluate_report", side_effect=[eval1, eval2]) as mock_critic, \
             patch.dict(os.environ, {"GROQ_API_KEY": "dummy_gsk", "TAVILY_API_KEY": "dummy_tvly"}):

            result = pipeline.run_research_pipeline("Early Approval Topic", max_refinements=2)

            # Stopped after 1 refinement because score reached 9.0
            self.assertEqual(result["refinement_attempts"], 1)
            self.assertEqual(mock_critic.call_count, 2)
            self.assertEqual(mock_writer.call_count, 2)

    # ── Test 8: End-to-End Workflow with 3 Distinct Topics ────────────────────
    def test_three_different_research_topics(self):
        topics = [
            "CRISPR gene editing clinical breakthroughs in 2025",
            "Quantum computing error mitigation architectures",
            "Autonomous AI agents in software engineering",
        ]

        for topic in topics:
            report_for_topic = ResearchReport(
                topic=topic,
                executive_summary=f"Executive summary of research on {topic}.",
                introduction=f"Introduction to the developments in {topic}.",
                key_findings=[
                    f"First critical advancement identified in {topic}.",
                    f"Second notable development with empirical support in {topic}.",
                    f"Third strategic implication for {topic}.",
                ],
                detailed_analysis=f"Detailed evidence and analysis regarding {topic}.",
                conclusion=f"Conclusion summarizing future outlook for {topic}.",
                references=["https://example.com/ref-a", "https://example.com/ref-b"],
            )

            approved_eval = CriticEvaluation(
                score=8.5,
                strengths=["Comprehensive findings", "Grounded analysis"],
                weak_explanations=[],
                missing_information=[],
                unsupported_claims=[],
                actionable_recommendations=[],
                needs_refinement=False,
                verdict=f"Approved report for {topic}",
            )

            with patch("pipeline.execute_search_agent", return_value=[{"title": "Source 1", "url": "https://example.com/ref-a", "snippet": "Snippet A"}]), \
                 patch("pipeline.execute_reader_agent", return_value=self.sample_sources), \
                 patch("pipeline.generate_report", return_value=report_for_topic), \
                 patch("pipeline.evaluate_report", return_value=approved_eval), \
                 patch.dict(os.environ, {"GROQ_API_KEY": "dummy_gsk", "TAVILY_API_KEY": "dummy_tvly"}):

                state = pipeline.run_research_pipeline(topic)
                self.assertEqual(state["topic"], topic)
                self.assertIn("## Executive Summary", state["report"])
                self.assertIn("## Key Findings", state["report"])
                self.assertIn("## References", state["report"])
                self.assertIn("8.5 / 10", state["critic"])
                self.assertTrue(len(state["sources"]) >= 1)

    # ── Test 9: Missing API Keys and Error Sanitization ───────────────────────
    def test_missing_keys_and_sanitization(self):
        # Missing keys raise ValueError
        with patch.dict(os.environ, {"GROQ_API_KEY": "", "TAVILY_API_KEY": ""}, clear=False):
            with self.assertRaises(ValueError) as ctx:
                pipeline.validate_environment()
            self.assertIn("GROQ_API_KEY", str(ctx.exception))
            self.assertIn("TAVILY_API_KEY", str(ctx.exception))

        # Sanitization redacts keys
        fake_groq_key = "gsk_test1234567890abcdefghijklmnopqrstuvwxyz"
        fake_tavily_key = "tvly-test1234567890abcdefghijklmnopqrstuvwxyz"
        error_msg = f"Failed to authenticate with {fake_groq_key} and {fake_tavily_key}"

        sanitized = tools.sanitize_error_message(error_msg)
        self.assertNotIn(fake_groq_key, sanitized)
        self.assertNotIn(fake_tavily_key, sanitized)
        self.assertIn("[REDACTED_API_KEY]", sanitized)


if __name__ == "__main__":
    unittest.main()

