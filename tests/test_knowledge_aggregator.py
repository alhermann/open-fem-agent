"""Tests for cross-session knowledge aggregation."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from core.session_journal import SessionJournal
from core.knowledge_aggregator import (
    AggregatedPattern,
    aggregate_sessions,
    normalize_error,
    save_aggregated,
)


class TestNormalizeError(unittest.TestCase):

    def test_numbers_replaced(self):
        self.assertEqual(normalize_error("expected 4 got 3"),
                         "expected num got num")

    def test_paths_replaced(self):
        self.assertIn("path", normalize_error("File /home/user/mesh.e not found"))

    def test_hex_addresses_replaced(self):
        self.assertIn("addr", normalize_error(
            "Object at 0x7f4e2c016fc0 failed"))

    def test_similar_errors_match(self):
        from difflib import SequenceMatcher
        a = normalize_error("expected 4 nodes but got 3 at line 42")
        b = normalize_error("expected 8 nodes but got 7 at line 99")
        ratio = SequenceMatcher(None, a, b).ratio()
        self.assertGreater(ratio, 0.85)


class TestAggregation(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.session_dir = Path(self.tmpdir) / "sessions"
        self.session_dir.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def _create_session(self, session_id, solver, error_msg):
        """Create a session with an error→success pattern."""
        j = SessionJournal(session_id=session_id)
        j.record("tool_error", "run_simulation", solver=solver,
                 error_message=error_msg)
        j.record("tool_success", "run_simulation", solver=solver)
        j.save(self.session_dir)

    def test_same_error_across_sessions_promoted(self):
        """Same error in 3 sessions → promoted."""
        self._create_session("s1", "fenics", "ValueError: bad mesh")
        self._create_session("s2", "fenics", "ValueError: bad mesh")
        self._create_session("s3", "fenics", "ValueError: bad mesh")

        patterns = aggregate_sessions(self.session_dir, min_sessions=3)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].session_count, 3)
        self.assertGreater(patterns[0].confidence, 0.7)

    def test_below_threshold_not_promoted(self):
        """Only 1 session → not promoted."""
        self._create_session("s1", "fenics", "ValueError: bad mesh")

        patterns = aggregate_sessions(self.session_dir, min_sessions=2)
        self.assertEqual(len(patterns), 0)

    def test_similar_errors_clustered(self):
        """Errors with different numbers but same structure cluster together."""
        self._create_session("s1", "fourc", "NUMDOF mismatch: expected 3 got 2")
        self._create_session("s2", "fourc", "NUMDOF mismatch: expected 4 got 3")

        patterns = aggregate_sessions(self.session_dir, min_sessions=2)
        self.assertEqual(len(patterns), 1)
        self.assertEqual(patterns[0].session_count, 2)

    def test_different_solvers_not_clustered(self):
        """Same error on different solvers stays separate."""
        self._create_session("s1", "fenics", "mesh error")
        self._create_session("s2", "fourc", "mesh error")

        patterns = aggregate_sessions(self.session_dir, min_sessions=2)
        self.assertEqual(len(patterns), 0)  # different solvers, 1 each

    def test_empty_dir(self):
        patterns = aggregate_sessions(self.session_dir, min_sessions=2)
        self.assertEqual(len(patterns), 0)

    def test_nonexistent_dir(self):
        patterns = aggregate_sessions(Path("/nonexistent"), min_sessions=2)
        self.assertEqual(len(patterns), 0)


class TestSaveAggregated(unittest.TestCase):

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        patterns = [
            AggregatedPattern("test", "pitfall", "fenics", "poisson",
                              "desc", 3, 0.85, ["s1", "s2", "s3"]),
        ]
        path = Path(self.tmpdir) / "aggregated.json"
        save_aggregated(patterns, path)

        data = json.loads(path.read_text())
        self.assertEqual(len(data["patterns"]), 1)
        self.assertEqual(data["patterns"][0]["session_count"], 3)

    def test_save_empty(self):
        path = Path(self.tmpdir) / "empty.json"
        save_aggregated([], path)
        data = json.loads(path.read_text())
        self.assertEqual(len(data["patterns"]), 0)


if __name__ == "__main__":
    unittest.main()
