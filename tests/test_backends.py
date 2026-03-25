#!/usr/bin/env python3
"""
Backend availability and template generation tests.

Verifies that:
1. All backends load and register correctly
2. Available backends pass their own availability check
3. Templates generate valid-looking input
4. Physics knowledge is non-empty for all supported modules
"""

import sys
import os
import unittest
from pathlib import Path

# Setup paths
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
os.environ.setdefault("FOURC_ROOT", str(Path(__file__).resolve().parents[2]))


class TestBackendRegistry(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from core.registry import load_all_backends, list_backends, _backends
        load_all_backends()
        cls.backends_info = list_backends()
        cls.backends = _backends

    def test_all_backends_registered(self):
        """All 4 backends should be registered."""
        names = {b["name"] for b in self.backends_info}
        self.assertIn("fourc", names)
        self.assertIn("fenics", names)
        self.assertIn("dealii", names)
        self.assertIn("febio", names)

    def test_backend_has_display_name(self):
        for b in self.backends_info:
            self.assertTrue(b["display_name"], f"{b['name']} has no display_name")

    def test_backend_has_physics(self):
        for b in self.backends_info:
            if b["status"] == "available":
                self.assertGreater(b["physics_count"], 0,
                                   f"{b['name']} has no physics modules")

    def test_backend_has_input_format(self):
        expected = {"fourc": "yaml", "fenics": "python", "dealii": "cpp", "febio": "xml"}
        for b in self.backends_info:
            if b["name"] in expected:
                self.assertEqual(b["input_format"], expected[b["name"]])


class TestFenicsBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fenics")

    def test_available(self):
        if not self.backend:
            self.skipTest("FEniCS not registered")
        status, msg = self.backend.check_availability()
        if status.value != "available":
            self.skipTest(f"FEniCS not available: {msg}")

    def test_poisson_template(self):
        if not self.backend:
            self.skipTest("FEniCS not registered")
        content = self.backend.generate_input("poisson", "2d", {"kappa": 2.5})
        self.assertIn("dolfinx", content)
        self.assertIn("2.5", content)
        self.assertIn("ufl", content)

    def test_elasticity_template(self):
        if not self.backend:
            self.skipTest("FEniCS not registered")
        content = self.backend.generate_input("linear_elasticity", "2d", {"E": 500})
        self.assertIn("500", content)
        self.assertIn("epsilon", content)

    def test_heat_template(self):
        if not self.backend:
            self.skipTest("FEniCS not registered")
        content = self.backend.generate_input("heat", "2d_steady", {"T_left": 200})
        self.assertIn("200", content)

    def test_knowledge(self):
        if not self.backend:
            self.skipTest("FEniCS not registered")
        k = self.backend.get_knowledge("poisson")
        self.assertIn("description", k)
        self.assertIn("pitfalls", k)

    def test_validate_valid(self):
        if not self.backend:
            self.skipTest("FEniCS not registered")
        content = self.backend.generate_input("poisson", "2d", {})
        errors = self.backend.validate_input(content)
        self.assertEqual(errors, [])


class TestDealiiBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("dealii")

    def test_poisson_template(self):
        if not self.backend:
            self.skipTest("deal.ii not registered")
        content = self.backend.generate_input("poisson", "2d", {"refinements": 3})
        self.assertIn("#include", content)
        self.assertIn("deal.II", content)
        self.assertIn("hyper_cube", content)

    def test_elasticity_template(self):
        if not self.backend:
            self.skipTest("deal.ii not registered")
        content = self.backend.generate_input("linear_elasticity", "2d", {})
        self.assertIn("FESystem", content)
        self.assertIn("1000.0", content)  # default E

    def test_validate_valid(self):
        if not self.backend:
            self.skipTest("deal.ii not registered")
        content = self.backend.generate_input("poisson", "2d", {})
        errors = self.backend.validate_input(content)
        self.assertEqual(errors, [])


class TestFourcBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("fourc")

    def test_has_physics(self):
        if not self.backend:
            self.skipTest("4C not registered")
        physics = self.backend.supported_physics()
        names = [p.name for p in physics]
        self.assertIn("poisson", names)
        self.assertIn("linear_elasticity", names)

    def test_availability(self):
        if not self.backend:
            self.skipTest("4C not registered")
        status, msg = self.backend.check_availability()
        # Don't assert AVAILABLE since 4C binary might not be present in CI
        self.assertIsNotNone(status)


class TestFebioBackend(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from core.registry import load_all_backends, get_backend
        load_all_backends()
        cls.backend = get_backend("febio")

    def test_has_physics(self):
        if not self.backend:
            self.skipTest("FEBio not registered")
        physics = self.backend.supported_physics()
        names = [p.name for p in physics]
        self.assertIn("linear_elasticity", names)

    def test_elasticity_template(self):
        if not self.backend:
            self.skipTest("FEBio not registered")
        content = self.backend.generate_input("linear_elasticity", "3d_cube", {"E": 2000})
        self.assertIn("febio_spec", content)
        self.assertIn("2000", content)

    def test_validate_valid(self):
        if not self.backend:
            self.skipTest("FEBio not registered")
        content = self.backend.generate_input("linear_elasticity", "3d_cube", {})
        errors = self.backend.validate_input(content)
        self.assertEqual(errors, [])


class TestSourceBuildDetection(unittest.TestCase):
    """Test that get_env_with_source_root correctly detects source builds."""

    def test_no_root_set(self):
        """When env var is not set, returns env unchanged."""
        from core.backend import get_env_with_source_root
        old = os.environ.pop("NONEXISTENT_ROOT", None)
        env = get_env_with_source_root("NONEXISTENT_ROOT")
        self.assertIsInstance(env, dict)
        # Should not crash, should return valid env
        self.assertIn("PATH", env)

    def test_root_set_no_build(self):
        """When root is set but has no build dir, root itself is added to PYTHONPATH."""
        from core.backend import get_env_with_source_root
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            os.environ["_TEST_ROOT"] = tmpdir
            env = get_env_with_source_root("_TEST_ROOT")
            self.assertIn(tmpdir, env.get("PYTHONPATH", ""))
            del os.environ["_TEST_ROOT"]

    def test_root_with_build_dir(self):
        """When root has build/ with .py files, build path prepended to PYTHONPATH."""
        from core.backend import get_env_with_source_root
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            build_dir = Path(tmpdir) / "build" / "lib"
            build_dir.mkdir(parents=True)
            (build_dir / "test_module.py").write_text("# test")
            os.environ["_TEST_ROOT"] = tmpdir
            env = get_env_with_source_root("_TEST_ROOT")
            pp = env.get("PYTHONPATH", "")
            # Build path should appear before root
            self.assertIn(str(build_dir), pp)
            del os.environ["_TEST_ROOT"]

    def test_dealii_cmake_hints(self):
        """deal.II cmake generator includes DEALII_ROOT hints."""
        from backends.dealii.backend import _generate_cmakelists
        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake build with cmake config
            cmake_dir = Path(tmpdir) / "build" / "lib" / "cmake" / "deal.II"
            cmake_dir.mkdir(parents=True)
            (cmake_dir / "deal.IIConfig.cmake").write_text("# fake")
            os.environ["DEALII_ROOT"] = tmpdir
            cmake = _generate_cmakelists("test_target")
            self.assertIn(str(cmake_dir), cmake)
            del os.environ["DEALII_ROOT"]

    def test_dealii_cmake_no_root(self):
        """Without DEALII_ROOT, cmake uses standard search paths."""
        from backends.dealii.backend import _generate_cmakelists
        old = os.environ.pop("DEALII_ROOT", None)
        cmake = _generate_cmakelists("test_target")
        self.assertIn("find_package(deal.II", cmake)
        self.assertIn("/usr", cmake)
        if old:
            os.environ["DEALII_ROOT"] = old

    def test_fourc_root_detected(self):
        """FOURC_ROOT is used by 4C backend for binary detection."""
        from backends.fourc.backend import _find_fourc_binary
        # Just verify it doesn't crash
        result = _find_fourc_binary()
        # May or may not find binary depending on system
        self.assertTrue(result is None or result.is_file())

    def test_developer_source_env_vars(self):
        """All developer entries have source_env_var field."""
        from tools.developer import _SOURCE_LOCATIONS
        for solver, info in _SOURCE_LOCATIONS.items():
            self.assertIn("source_env_var", info,
                          f"{solver} missing source_env_var in developer info")
            self.assertTrue(info["source_env_var"].endswith("_ROOT"),
                            f"{solver} env var should end with _ROOT")


if __name__ == "__main__":
    unittest.main()
