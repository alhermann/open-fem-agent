"""Generator registry for FEniCSx physics modules.

Each generator module exposes:
  - ``generate(variant, params) -> str`` — returns a runnable FEniCSx script
  - ``KNOWLEDGE`` — dict of domain knowledge for LLM consumption
  - ``VARIANTS`` — list of available variant names

Usage::

    from src.backends.fenics.generators import get_generator, list_all_physics

    script = get_generator("poisson", "2d")({})
    all_physics = list_all_physics()
"""

from __future__ import annotations

import importlib
import logging
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Maps physics name -> module name (relative import path within this package).
_PHYSICS_MODULES: dict[str, str] = {
    "poisson":              ".poisson",
    "linear_elasticity":    ".elasticity",
    "heat":                 ".heat",
    "navier_stokes":        ".navier_stokes",
    "stokes":               ".stokes",
    "hyperelasticity":      ".hyperelasticity",
    "thermal_structural":   ".thermal_structural",
    "convection_diffusion": ".convection_diffusion",
    "eigenvalue":           ".eigenvalue",
    "biharmonic":           ".biharmonic",
    "mixed_poisson":        ".mixed_poisson",
    "reaction_diffusion":   ".reaction_diffusion",
}

# Cache of imported modules.
_MODULE_CACHE: dict[str, Any] = {}


def _load_module(physics: str) -> Any:
    """Lazily import a generator module by physics name."""
    if physics in _MODULE_CACHE:
        return _MODULE_CACHE[physics]

    module_path = _PHYSICS_MODULES.get(physics)
    if module_path is None:
        raise KeyError(
            f"Unknown FEniCS physics: {physics!r}. "
            f"Available: {sorted(_PHYSICS_MODULES)}"
        )

    try:
        mod = importlib.import_module(module_path, package=__name__)
    except ImportError as exc:
        raise ImportError(
            f"Cannot import FEniCS generator module {module_path} "
            f"for physics {physics!r}: {exc}"
        ) from exc

    _MODULE_CACHE[physics] = mod
    return mod


def get_generator(physics: str, variant: str) -> Callable[[dict], str]:
    """Return a callable ``generator(params) -> script_str`` for the given physics+variant.

    Parameters
    ----------
    physics : str
        Physics name, e.g. ``"poisson"``, ``"linear_elasticity"``.
    variant : str
        Variant name, e.g. ``"2d"``, ``"3d"``, ``"l_domain"``.

    Returns
    -------
    Callable[[dict], str]
        A function that takes a params dict and returns a runnable script.

    Raises
    ------
    KeyError
        If *physics* is unknown.
    ValueError
        If *variant* is unknown for the given physics.
    """
    mod = _load_module(physics)
    # Return a partial that binds the variant
    def _gen(params: dict) -> str:
        return mod.generate(variant, params)
    return _gen


def generate_script(physics: str, variant: str, params: dict) -> str:
    """Generate a FEniCSx script for the given physics, variant, and parameters.

    Convenience wrapper around :func:`get_generator`.
    """
    mod = _load_module(physics)
    return mod.generate(variant, params)


def get_knowledge(physics: str) -> dict:
    """Return domain knowledge for the given physics.

    Parameters
    ----------
    physics : str
        Physics name.

    Returns
    -------
    dict
        Domain knowledge dictionary.
    """
    mod = _load_module(physics)
    return getattr(mod, "KNOWLEDGE", {})


def get_variants(physics: str) -> list[str]:
    """Return the list of available variants for a physics module."""
    mod = _load_module(physics)
    return getattr(mod, "VARIANTS", [])


def list_all_physics() -> list[str]:
    """Return sorted list of all registered physics names."""
    return sorted(_PHYSICS_MODULES.keys())


# General FEniCS knowledge (not tied to a specific physics module).
GENERAL_KNOWLEDGE = {
    "description": "FEniCSx (dolfinx) general capabilities",
    "element_families": {
        "Lagrange (P/Q)": "Continuous, arbitrary order, all cell types",
        "DG": "Discontinuous Lagrange, order 0+",
        "Raviart-Thomas": "H(div) conforming, for mixed methods",
        "BDM": "H(div) conforming, full polynomial",
        "Nedelec 1st kind": "H(curl) conforming, for Maxwell/electromagnetics",
        "Nedelec 2nd kind": "H(curl) conforming, full polynomial",
        "Crouzeix-Raviart": "Nonconforming, order 1 only",
        "Bubble": "For MINI element enrichment",
        "Hermite": "C1 conforming on simplices",
        "Serendipity": "Quad/hex only, fewer DOFs",
        "Regge": "For elasticity complexes",
    },
    "mesh_types": [
        "create_unit_square, create_unit_cube, create_box, create_rectangle",
        "Gmsh: gmshio.model_to_mesh (2D/3D, mixed cell types)",
        "XDMF import/export, refinement (refine, plaza_refine)",
    ],
    "solver_catalogue": {
        "direct": "MUMPS, SuperLU_dist, UMFPACK (via PETSc)",
        "iterative": "CG, GMRES, BiCGStab, MinRes, Richardson",
        "preconditioners": "ILU, ICC, Jacobi, SOR, GAMG, hypre/BoomerAMG, BDDC, fieldsplit",
        "nonlinear": "PETSc SNES (newtonls, newtontr, ngmres)",
        "eigenvalue": "SLEPc EPS (krylovschur, arnoldi, lanczos)",
    },
    "unique_features": [
        "UFL: symbolic weak form language with automatic differentiation",
        "PETSc/SLEPc: industrial-strength solver infrastructure",
        "Gmsh integration: complex geometry meshing via Python API",
        "MPI parallel: distributed assembly + solve out-of-box",
        "Complex-valued problems: complex PETSc build",
        "Mixed elements: arbitrary combinations via mixed_element()",
        "Checkpointing via adios4dolfinx",
    ],
}
