"""Tests for session analyzer — pattern detection, filtering, formatting."""

import json
import tempfile
import shutil
import unittest
from pathlib import Path

from core.session_journal import SessionJournal
from core.session_analyzer import (
    CandidateKnowledge,
    analyze_journal,
    analyze_journal_file,
    filter_against_existing,
    format_candidates,
)


def _make_journal_with_error_then_success():
    """Create a journal with a classic error→success pattern."""
    j = SessionJournal(session_id="test_ets")
    j.record("tool_call", "run_simulation", solver="fourc")
    j.record("tool_error", "run_simulation", solver="fourc",
             error_message="NUMDOF mismatch: expected 3, got 2")
    j.record("knowledge_lookup", "knowledge", solver="fourc",
             physics="fsi", notes="topic=pitfalls")
    j.record("tool_call", "run_simulation", solver="fourc")
    j.record("tool_success", "run_simulation", solver="fourc")
    return j


def _make_journal_with_source_reading():
    """Create a journal where agent reads source code."""
    j = SessionJournal(session_id="test_src")
    j.record("tool_call", "prepare_simulation", solver="fourc", physics="tsi")
    j.record("source_read", "developer", solver="fourc",
             notes="keyword=CLONING")
    j.record("tool_call", "run_with_generator", solver="fourc")
    j.record("tool_success", "run_with_generator", solver="fourc")
    return j


def _make_journal_with_convergence_issue():
    """Create a journal with a convergence issue."""
    j = SessionJournal(session_id="test_conv")
    j.record("tool_call", "coupled_solve", solver="fenics->fourc",
             physics="heat_dd")
    j.record("convergence_issue", "coupled_solve", solver="fenics->fourc",
             physics="heat_dd",
             details={"iterations": 15, "relaxation": 0.3})
    j.record("tool_success", "coupled_solve", solver="fenics->fourc")
    return j


def _make_clean_journal():
    """Create a journal with no issues — all smooth."""
    j = SessionJournal(session_id="test_clean")
    j.record("tool_call", "prepare_simulation", solver="fenics", physics="poisson")
    j.record("knowledge_lookup", "knowledge", solver="fenics", physics="poisson")
    j.record("tool_call", "run_simulation", solver="fenics")
    j.record("tool_success", "run_simulation", solver="fenics")
    return j


class TestPatternDetection(unittest.TestCase):
    """Test each detection pattern individually."""

    def test_error_then_success_detected(self):
        j = _make_journal_with_error_then_success()
        candidates = analyze_journal(j)
        pitfalls = [c for c in candidates if c.category == "pitfall"]
        self.assertGreaterEqual(len(pitfalls), 1)
        self.assertIn("NUMDOF", pitfalls[0].title)

    def test_knowledge_after_failure_detected(self):
        j = _make_journal_with_error_then_success()
        candidates = analyze_journal(j)
        doc_gaps = [c for c in candidates if c.category == "doc_gap"]
        self.assertGreaterEqual(len(doc_gaps), 1)

    def test_source_reading_detected(self):
        j = _make_journal_with_source_reading()
        candidates = analyze_journal(j)
        doc_gaps = [c for c in candidates if c.category == "doc_gap"]
        self.assertGreaterEqual(len(doc_gaps), 1)
        self.assertIn("CLONING", doc_gaps[0].title)

    def test_convergence_issue_detected(self):
        j = _make_journal_with_convergence_issue()
        candidates = analyze_journal(j)
        tips = [c for c in candidates if c.category == "parameter_tip"]
        self.assertGreaterEqual(len(tips), 1)
        self.assertIn("15", tips[0].title)

    def test_clean_session_no_candidates(self):
        j = _make_clean_journal()
        candidates = analyze_journal(j)
        # Clean session may have knowledge_lookup but no errors → no pitfalls
        pitfalls = [c for c in candidates if c.category == "pitfall"]
        self.assertEqual(len(pitfalls), 0)

    def test_empty_journal_no_candidates(self):
        j = SessionJournal(session_id="empty")
        candidates = analyze_journal(j)
        self.assertEqual(len(candidates), 0)


class TestConfidence(unittest.TestCase):
    """Test confidence scoring."""

    def test_direct_fix_higher_confidence(self):
        """Error immediately followed by success → high confidence."""
        j = SessionJournal(session_id="direct")
        j.record("tool_error", "run_simulation", solver="x", error_message="fail")
        j.record("tool_success", "run_simulation", solver="x")
        candidates = analyze_journal(j)
        pitfalls = [c for c in candidates if c.category == "pitfall"]
        self.assertGreaterEqual(len(pitfalls), 1)
        self.assertGreater(pitfalls[0].confidence, 0.7)

    def test_many_steps_lower_confidence(self):
        """Error with many steps before success → lower confidence."""
        j = SessionJournal(session_id="slow")
        j.record("tool_error", "run_simulation", solver="x", error_message="fail")
        for _ in range(10):
            j.record("tool_call", "knowledge", solver="x")
        j.record("tool_success", "run_simulation", solver="x")
        candidates = analyze_journal(j)
        pitfalls = [c for c in candidates if c.category == "pitfall"]
        self.assertGreaterEqual(len(pitfalls), 1)
        self.assertLess(pitfalls[0].confidence, 0.7)

    def test_source_reading_low_confidence(self):
        j = _make_journal_with_source_reading()
        candidates = analyze_journal(j)
        doc_gaps = [c for c in candidates if c.category == "doc_gap"]
        for c in doc_gaps:
            self.assertLessEqual(c.confidence, 0.5)


class TestFiltering(unittest.TestCase):
    """Test filtering against existing knowledge."""

    def test_known_pitfall_filtered(self):
        j = _make_journal_with_error_then_success()
        candidates = analyze_journal(j)
        existing = ["NUMDOF mismatch: expected 3, got 2 in multi-physics"]
        filtered = filter_against_existing(candidates, existing)
        pitfalls = [c for c in filtered if c.category == "pitfall"]
        self.assertEqual(len(pitfalls), 0)

    def test_novel_pitfall_kept(self):
        j = _make_journal_with_error_then_success()
        candidates = analyze_journal(j)
        existing = ["completely unrelated pitfall about mesh generation"]
        filtered = filter_against_existing(candidates, existing)
        pitfalls = [c for c in filtered if c.category == "pitfall"]
        self.assertGreaterEqual(len(pitfalls), 1)

    def test_low_confidence_filtered(self):
        j = _make_journal_with_source_reading()
        candidates = analyze_journal(j)
        filtered = filter_against_existing(candidates, [], threshold=0.5)
        # Source reading has confidence 0.4 → should be filtered
        self.assertEqual(len(filtered), 0)

    def test_empty_existing_keeps_all(self):
        j = _make_journal_with_error_then_success()
        candidates = analyze_journal(j)
        filtered = filter_against_existing(candidates, [], threshold=0.0)
        self.assertEqual(len(filtered), len(candidates))


class TestDeduplication(unittest.TestCase):
    """Test that near-duplicate candidates are merged."""

    def test_duplicate_errors_merged(self):
        j = SessionJournal(session_id="dup")
        # Same error twice
        j.record("tool_error", "run_simulation", solver="x",
                 error_message="mesh too coarse for convergence")
        j.record("tool_success", "run_simulation", solver="x")
        j.record("tool_error", "run_simulation", solver="x",
                 error_message="mesh too coarse for convergence")
        j.record("tool_success", "run_simulation", solver="x")
        candidates = analyze_journal(j)
        pitfalls = [c for c in candidates if c.category == "pitfall"]
        # Should be deduplicated to 1
        self.assertEqual(len(pitfalls), 1)


class TestFormatting(unittest.TestCase):
    """Test human-readable formatting."""

    def test_format_empty(self):
        result = format_candidates([])
        self.assertIn("No knowledge", result)

    def test_format_single(self):
        c = CandidateKnowledge(
            category="pitfall", solver="fourc", physics="fsi",
            title="Test pitfall", description="Details here",
            confidence=0.8,
        )
        result = format_candidates([c])
        self.assertIn("PITFALL", result)
        self.assertIn("Test pitfall", result)
        self.assertIn("0.8", result)

    def test_format_multiple(self):
        candidates = [
            CandidateKnowledge("pitfall", "a", "", "P1", "d", 0.9),
            CandidateKnowledge("doc_gap", "b", "", "D1", "d", 0.5),
        ]
        result = format_candidates(candidates)
        self.assertIn("[1]", result)
        self.assertIn("[2]", result)


class TestSerialisation(unittest.TestCase):
    """Test save/load of candidates and journal file analysis."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_candidate_to_dict(self):
        c = CandidateKnowledge("pitfall", "fourc", "fsi", "t", "d", 0.8)
        d = c.to_dict()
        self.assertEqual(d["category"], "pitfall")
        self.assertEqual(d["confidence"], 0.8)

    def test_analyze_from_file(self):
        j = _make_journal_with_error_then_success()
        path = j.save(self.tmpdir)
        candidates = analyze_journal_file(path)
        self.assertGreater(len(candidates), 0)

    def test_combined_journal_all_patterns(self):
        """Journal with all 4 patterns — should detect all."""
        j = SessionJournal(session_id="combined")
        # Pattern 1: error→success
        j.record("tool_error", "run_simulation", solver="fourc",
                 error_message="missing section")
        j.record("tool_success", "run_simulation", solver="fourc")
        # Pattern 2: error→knowledge
        j.record("tool_error", "run_with_generator", solver="dealii",
                 error_message="cmake failed")
        j.record("knowledge_lookup", "knowledge", solver="dealii",
                 notes="topic=pitfalls")
        # Pattern 3: source read
        j.record("source_read", "developer", solver="fourc",
                 notes="keyword=TSI")
        # Pattern 4: convergence
        j.record("convergence_issue", "coupled_solve", solver="fenics->ngsolve",
                 details={"iterations": 20, "relaxation": 0.2})

        candidates = analyze_journal(j)
        categories = {c.category for c in candidates}
        self.assertIn("pitfall", categories)
        self.assertIn("doc_gap", categories)
        self.assertIn("parameter_tip", categories)


if __name__ == "__main__":
    unittest.main()
