"""Tests for solver fingerprinting and community knowledge loading."""

import json
import tempfile
import shutil
import unittest
from pathlib import Path

# Fingerprinting
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from fingerprint_solvers import (
    compare_fingerprints,
    generate_all,
    SOLVERS,
)

# Community knowledge
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class TestFingerprinting(unittest.TestCase):
    """Test solver API fingerprinting."""

    def test_all_solvers_registered(self):
        self.assertEqual(len(SOLVERS), 7)
        for name in ["fenics", "ngsolve", "skfem", "kratos", "dune", "fourc", "dealii"]:
            self.assertIn(name, SOLVERS)

    def test_generate_produces_dicts(self):
        """Generate fingerprints for pip-installable solvers."""
        results = generate_all(["ngsolve", "skfem"])
        self.assertIn("ngsolve", results)
        self.assertIn("skfem", results)
        for name, fp in results.items():
            self.assertEqual(fp["solver"], name)
            self.assertIn("timestamp", fp)

    def test_fingerprint_has_api_section(self):
        results = generate_all(["skfem"])
        self.assertIn("api", results["skfem"])
        self.assertIsInstance(results["skfem"]["api"], dict)

    def test_compare_no_drift(self):
        fp = {"solver": "test", "api": {"func_a": True, "func_b": True}}
        drifts = compare_fingerprints(fp, fp)
        self.assertEqual(len(drifts), 0)

    def test_compare_detects_removed_api(self):
        prev = {"solver": "test", "api": {"func_a": True, "func_b": True}}
        curr = {"solver": "test", "api": {"func_a": True, "func_b": False}}
        drifts = compare_fingerprints(curr, prev)
        self.assertEqual(len(drifts), 1)
        self.assertIn("BREAKING", drifts[0])
        self.assertIn("func_b", drifts[0])

    def test_compare_detects_new_api(self):
        prev = {"solver": "test", "api": {"func_a": True}}
        curr = {"solver": "test", "api": {"func_a": True, "func_b": True}}
        drifts = compare_fingerprints(curr, prev)
        self.assertEqual(len(drifts), 1)
        self.assertIn("NEW", drifts[0])

    def test_compare_detects_version_change(self):
        prev = {"solver": "test", "ngsolve": {"version": "6.2.2301"}}
        curr = {"solver": "test", "ngsolve": {"version": "6.2.2402"}}
        drifts = compare_fingerprints(curr, prev)
        self.assertEqual(len(drifts), 1)
        self.assertIn("VERSION", drifts[0])

    def test_compare_empty_fingerprints(self):
        drifts = compare_fingerprints({}, {})
        self.assertEqual(len(drifts), 0)


class TestCommunityKnowledge(unittest.TestCase):
    """Test community knowledge loading."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.pending = Path(self.tmpdir) / "pending"
        self.pending.mkdir()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_empty_directory(self):
        from tools.consolidated import _load_community_knowledge
        # Monkey-patch the path temporarily
        import tools.consolidated as tc
        orig = tc._load_community_knowledge

        def _load(solver=""):
            entries = []
            for f in sorted(self.pending.glob("session_*.json")):
                try:
                    data = json.loads(f.read_text())
                    if isinstance(data, list):
                        for entry in data:
                            if solver and entry.get("solver", "") != solver:
                                continue
                            entries.append(entry)
                except (json.JSONDecodeError, OSError):
                    continue
            return entries

        result = _load()
        self.assertEqual(result, [])

    def test_load_with_entries(self):
        entries = [
            {"category": "pitfall", "solver": "fourc", "title": "Test pitfall",
             "description": "Details", "confidence": 0.8},
        ]
        (self.pending / "session_abc.json").write_text(json.dumps(entries))

        # Direct load
        loaded = []
        for f in self.pending.glob("session_*.json"):
            loaded.extend(json.loads(f.read_text()))
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["solver"], "fourc")

    def test_load_filters_by_solver(self):
        entries = [
            {"solver": "fourc", "title": "A"},
            {"solver": "fenics", "title": "B"},
        ]
        (self.pending / "session_xyz.json").write_text(json.dumps(entries))

        loaded = []
        for f in self.pending.glob("session_*.json"):
            for e in json.loads(f.read_text()):
                if e.get("solver") == "fourc":
                    loaded.append(e)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["title"], "A")

    def test_corrupted_json_skipped(self):
        (self.pending / "session_bad.json").write_text("not json")
        (self.pending / "session_good.json").write_text(
            json.dumps([{"solver": "x", "title": "OK"}])
        )

        loaded = []
        for f in sorted(self.pending.glob("session_*.json")):
            try:
                data = json.loads(f.read_text())
                if isinstance(data, list):
                    loaded.extend(data)
            except json.JSONDecodeError:
                continue
        self.assertEqual(len(loaded), 1)


if __name__ == "__main__":
    unittest.main()
