"""Tests for solver auto-discovery."""

import json
import os
import tempfile
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch

from core.autodiscovery import (
    ProbeResult,
    discover_backends,
    format_discovery,
    save_discovered_config,
    load_discovered_config,
    _probe_pip_package,
    _get_git_info,
)


class TestProbeResult(unittest.TestCase):
    def test_to_dict_drops_empty(self):
        r = ProbeResult(backend="test", found=True, confidence="definite",
                        location="/usr/bin/test")
        d = r.to_dict()
        self.assertNotIn("version", d)  # empty string dropped
        self.assertIn("location", d)

    def test_not_found_has_install_hint(self):
        r = ProbeResult(backend="test", found=False,
                        install_hint="pip install test")
        self.assertIn("pip", r.install_hint)


class TestPipProbe(unittest.TestCase):
    def test_probe_installed_package(self):
        # numpy is always available
        r = _probe_pip_package("numpy", "numpy", "test_np", "pip install numpy")
        self.assertTrue(r.found)
        self.assertEqual(r.confidence, "definite")
        self.assertIn(".", r.version)  # has a version string

    def test_probe_missing_package(self):
        r = _probe_pip_package("nonexistent_pkg_xyz", "nonexistent_pkg_xyz",
                               "test_missing", "pip install nonexistent_pkg_xyz")
        self.assertFalse(r.found)
        self.assertEqual(r.confidence, "not_found")


class TestDiscoverBackends(unittest.TestCase):
    def test_returns_list_of_probes(self):
        results = discover_backends()
        self.assertIsInstance(results, list)
        self.assertGreater(len(results), 0)
        for r in results:
            self.assertIsInstance(r, ProbeResult)

    def test_at_least_some_found(self):
        """At least pip-installable solvers should be found."""
        results = discover_backends()
        found = [r for r in results if r.found]
        self.assertGreater(len(found), 0,
                           "Expected at least one solver to be found")

    def test_all_7_backends_probed(self):
        results = discover_backends()
        backends = {r.backend for r in results}
        for expected in ["ngsolve", "skfem", "kratos", "dune",
                         "fenics", "fourc", "dealii"]:
            self.assertIn(expected, backends,
                          f"Backend {expected} not probed")


class TestFormatDiscovery(unittest.TestCase):
    def test_format_with_found_and_missing(self):
        results = [
            ProbeResult("fenics", True, "definite", "/usr/bin/python", "0.9.0"),
            ProbeResult("fourc", False, "not_found", install_hint="Build from source"),
        ]
        text = format_discovery(results)
        self.assertIn("1 found", text)
        self.assertIn("1 missing", text)
        self.assertIn("fenics", text)
        self.assertIn("fourc", text)

    def test_format_empty(self):
        text = format_discovery([])
        self.assertIn("0 found", text)


class TestPersistConfig(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self._orig_path = __import__("core.autodiscovery",
                                      fromlist=["CONFIG_PATH"]).CONFIG_PATH

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        import core.autodiscovery
        core.autodiscovery.CONFIG_PATH = self._orig_path

    def test_save_and_load_round_trip(self):
        import core.autodiscovery
        core.autodiscovery.CONFIG_PATH = Path(self.tmpdir) / "config.json"

        results = [
            ProbeResult("fenics", True, "definite", "/opt/python", "0.9.0"),
            ProbeResult("fourc", True, "definite", "/opt/4C/build/4C"),
        ]
        save_discovered_config(results)
        config = load_discovered_config()
        self.assertIsNotNone(config)
        self.assertIn("fenics", config["backends"])
        self.assertIn("fourc", config["backends"])
        self.assertEqual(config["backends"]["fenics"]["version"], "0.9.0")

    def test_load_nonexistent_returns_none(self):
        import core.autodiscovery
        core.autodiscovery.CONFIG_PATH = Path(self.tmpdir) / "nonexistent.json"
        self.assertIsNone(load_discovered_config())


class TestGitInfo(unittest.TestCase):
    def test_git_info_on_this_repo(self):
        """This repo itself is a git repo — we can probe it."""
        repo = str(Path(__file__).parent.parent)
        info = _get_git_info(repo)
        if info:  # might be None if git not installed
            self.assertIn("branch", info)
            self.assertIsInstance(info.get("uncommitted_changes", 0), int)

    def test_git_info_on_nonexistent_dir(self):
        info = _get_git_info("/nonexistent/path")
        self.assertIsNone(info)


if __name__ == "__main__":
    unittest.main()
