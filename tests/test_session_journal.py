"""Tests for the session journal — event recording, serialisation, edge cases."""

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from core.session_journal import (
    EVENT_TYPES,
    JournalEvent,
    SessionJournal,
    get_journal,
    reset_journal,
)


class TestJournalEvent(unittest.TestCase):
    """Test JournalEvent dataclass."""

    def test_create_valid_event(self):
        evt = JournalEvent(
            timestamp=time.time(),
            event_type="tool_call",
            tool_name="run_simulation",
            solver="fourc",
        )
        self.assertEqual(evt.event_type, "tool_call")
        self.assertEqual(evt.solver, "fourc")

    def test_invalid_event_type_raises(self):
        with self.assertRaises(ValueError):
            JournalEvent(
                timestamp=time.time(),
                event_type="bogus_type",
                tool_name="test",
            )

    def test_all_event_types_valid(self):
        for et in EVENT_TYPES:
            evt = JournalEvent(timestamp=0, event_type=et, tool_name="t")
            self.assertEqual(evt.event_type, et)

    def test_long_error_message_truncated(self):
        msg = "x" * 1000
        evt = JournalEvent(
            timestamp=0, event_type="tool_error", tool_name="t",
            error_message=msg,
        )
        self.assertLessEqual(len(evt.error_message), 500)
        self.assertTrue(evt.error_message.endswith("..."))

    def test_long_notes_truncated(self):
        notes = "n" * 1000
        evt = JournalEvent(
            timestamp=0, event_type="tool_call", tool_name="t",
            notes=notes,
        )
        self.assertLessEqual(len(evt.notes), 500)

    def test_non_dict_details_replaced(self):
        evt = JournalEvent(
            timestamp=0, event_type="tool_call", tool_name="t",
            details="not a dict",
        )
        self.assertEqual(evt.details, {})

    def test_unicode_in_fields(self):
        evt = JournalEvent(
            timestamp=0, event_type="tool_call", tool_name="t",
            solver="Ünïcödé", notes="日本語テスト",
        )
        self.assertEqual(evt.solver, "Ünïcödé")


class TestSessionJournal(unittest.TestCase):
    """Test SessionJournal recording and queries."""

    def setUp(self):
        self.journal = SessionJournal()

    def test_record_returns_event(self):
        evt = self.journal.record("tool_call", "run_simulation", solver="fenics")
        self.assertIsInstance(evt, JournalEvent)
        self.assertEqual(len(self.journal.events), 1)

    def test_record_invalid_type_returns_none(self):
        evt = self.journal.record("invalid_type", "test")
        self.assertIsNone(evt)
        self.assertEqual(len(self.journal.events), 0)

    def test_solvers_and_physics_tracked(self):
        self.journal.record("tool_call", "t", solver="fenics", physics="poisson")
        self.journal.record("tool_call", "t", solver="fourc", physics="fsi")
        self.journal.record("tool_call", "t", solver="fenics", physics="heat")
        self.assertEqual(self.journal.solvers_used, {"fenics", "fourc"})
        self.assertEqual(self.journal.physics_used, {"poisson", "fsi", "heat"})

    def test_error_count(self):
        self.journal.record("tool_call", "t")
        self.journal.record("tool_error", "t", error_message="fail")
        self.journal.record("tool_success", "t")
        self.journal.record("tool_error", "t", error_message="fail2")
        self.assertEqual(self.journal.error_count, 2)

    def test_events_by_type(self):
        self.journal.record("tool_call", "a")
        self.journal.record("tool_error", "b")
        self.journal.record("tool_call", "c")
        calls = self.journal.events_by_type("tool_call")
        self.assertEqual(len(calls), 2)

    def test_duration_empty_journal(self):
        self.assertEqual(self.journal.duration_seconds, 0.0)

    def test_duration_with_events(self):
        self.journal.record("tool_call", "t")
        time.sleep(0.05)
        self.journal.record("tool_call", "t")
        self.assertGreater(self.journal.duration_seconds, 0.0)

    def test_reset(self):
        self.journal.record("tool_call", "t", solver="x", physics="y")
        self.journal.reset()
        self.assertEqual(len(self.journal.events), 0)
        self.assertEqual(self.journal.solvers_used, set())
        self.assertEqual(self.journal.physics_used, set())

    def test_record_never_raises_on_bad_input(self):
        """record() must never crash the tool it instruments."""
        # None values
        evt = self.journal.record("tool_call", "t", solver=None)
        # This should not raise — solver gets str(None)
        self.assertIsNotNone(evt)

    def test_many_events(self):
        """Stress test: 10k events should work fine."""
        for i in range(10_000):
            self.journal.record("tool_call", f"tool_{i}", solver="s")
        self.assertEqual(len(self.journal.events), 10_000)


class TestJournalSerialisation(unittest.TestCase):
    """Test save/load round-trip."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.journal = SessionJournal(session_id="test123")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_to_dict(self):
        self.journal.record("tool_call", "run_simulation", solver="fenics")
        d = self.journal.to_dict()
        self.assertEqual(d["session_id"], "test123")
        self.assertEqual(d["event_count"], 1)
        self.assertIn("events", d)
        self.assertIn("solvers_used", d)

    def test_save_creates_file(self):
        self.journal.record("tool_call", "t")
        path = self.journal.save(self.tmpdir)
        self.assertTrue(path.exists())
        self.assertEqual(path.name, "session_test123.json")

    def test_save_creates_directory(self):
        nested = os.path.join(self.tmpdir, "a", "b", "c")
        path = self.journal.save(nested)
        self.assertTrue(path.exists())

    def test_save_valid_json(self):
        self.journal.record("tool_error", "t", error_message="boom",
                           details={"key": "value"})
        path = self.journal.save(self.tmpdir)
        data = json.loads(path.read_text())
        self.assertEqual(data["session_id"], "test123")
        self.assertEqual(len(data["events"]), 1)

    def test_save_load_round_trip(self):
        self.journal.record("tool_call", "run_simulation",
                           solver="fenics", physics="poisson")
        self.journal.record("tool_error", "run_simulation",
                           solver="fenics", error_message="mesh too coarse")
        self.journal.record("knowledge_lookup", "knowledge",
                           solver="fenics", physics="poisson",
                           notes="looked up pitfalls")
        path = self.journal.save(self.tmpdir)

        loaded = SessionJournal.load(path)
        self.assertEqual(loaded.session_id, "test123")
        self.assertEqual(len(loaded.events), 3)
        self.assertEqual(loaded.solvers_used, {"fenics"})
        self.assertEqual(loaded.physics_used, {"poisson"})
        self.assertEqual(loaded.error_count, 1)

    def test_save_empty_journal(self):
        path = self.journal.save(self.tmpdir)
        data = json.loads(path.read_text())
        self.assertEqual(data["event_count"], 0)
        self.assertEqual(data["events"], [])

    def test_save_with_special_characters(self):
        self.journal.record("tool_error", "t",
                           error_message='Error: "quotes" & <brackets> \n\ttabs')
        path = self.journal.save(self.tmpdir)
        loaded = SessionJournal.load(path)
        self.assertIn("quotes", loaded.events[0].error_message)

    def test_to_dict_drops_empty_fields(self):
        self.journal.record("tool_call", "t")
        d = self.journal.to_dict()
        evt = d["events"][0]
        # Empty strings should be dropped
        self.assertNotIn("solver", evt)
        self.assertNotIn("error_message", evt)
        # Non-empty should be present
        self.assertIn("tool_name", evt)

    def test_load_corrupted_file_raises(self):
        path = Path(self.tmpdir) / "bad.json"
        path.write_text("not json at all")
        with self.assertRaises(json.JSONDecodeError):
            SessionJournal.load(path)


class TestSingleton(unittest.TestCase):
    """Test singleton pattern."""

    def test_get_journal_returns_same_instance(self):
        j1 = get_journal()
        j2 = get_journal()
        self.assertIs(j1, j2)

    def test_reset_journal_creates_new_instance(self):
        j1 = get_journal()
        j2 = reset_journal()
        self.assertIsNot(j1, j2)
        j3 = get_journal()
        self.assertIs(j2, j3)


if __name__ == "__main__":
    unittest.main()
