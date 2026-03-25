"""Generator registry for deal.II physics templates.

Each generator module exposes:
  - A template function  ``generate(params) -> str``
  - A knowledge dict     ``KNOWLEDGE: dict``
  - A template key list  ``TEMPLATE_KEYS: list[str]``  (physics_variant keys it provides)

The registry is populated lazily so missing modules don't prevent startup.
"""

from __future__ import annotations

import importlib
import logging
from typing import Callable

logger = logging.getLogger(__name__)

# Maps template key ("physics_variant") to (module_path, function_name).
_TEMPLATE_SPECS: dict[str, tuple[str, str]] = {
    # poisson
    "poisson_2d":                (".poisson",              "_poisson_2d"),
    "poisson_3d":                (".poisson",              "_poisson_3d"),
    "poisson_l_domain":          (".poisson",              "_poisson_l_domain"),
    "poisson_rectangle":         (".poisson",              "_poisson_rectangle"),
    "poisson_2d_adaptive":       (".poisson",              "_poisson_adaptive_2d"),
    # elasticity
    "linear_elasticity_2d":      (".elasticity",           "_elasticity_2d"),
    "linear_elasticity_thick_beam": (".elasticity",        "_elasticity_thick_beam"),
    # heat
    "heat_2d_transient":         (".heat",                 "_heat_2d_transient"),
    "heat_2d_steady":            (".heat",                 "_heat_2d_steady"),
    "heat_rectangle":            (".heat",                 "_heat_rectangle"),
    # stokes
    "stokes_2d":                 (".stokes",               "_stokes_2d"),
    # convection diffusion
    "convection_diffusion_2d":   (".convection_diffusion", "_convdiff_2d"),
    # nonlinear
    "nonlinear_2d_minimal_surface": (".nonlinear",         "_nonlinear_minimal_surface_2d"),
    # helmholtz
    "helmholtz_2d":              (".helmholtz",            "_helmholtz_2d"),
    # eigenvalue
    "eigenvalue_2d":             (".eigenvalue",           "_eigenvalue_2d"),
    # wave
    "wave_2d":                   (".wave",                 "_wave_2d"),
    # hp adaptive
    "hp_adaptive_2d":            (".hp_adaptive",          "_hp_adaptive_2d"),
    # dg transport
    "dg_transport_2d":           (".dg_transport",         "_dg_transport_2d"),
    # hyperelasticity
    "hyperelasticity_3d":        (".hyperelasticity",      "_hyperelasticity_3d"),
    # parallel
    "parallel_poisson_2d":       (".parallel",             "_parallel_poisson_2d"),
}

# Maps physics name to (module_path, dict_name) for knowledge.
_KNOWLEDGE_SPECS: dict[str, tuple[str, str]] = {
    "poisson":              (".poisson",              "KNOWLEDGE"),
    "linear_elasticity":    (".elasticity",           "KNOWLEDGE"),
    "heat":                 (".heat",                 "KNOWLEDGE"),
    "stokes":               (".stokes",               "KNOWLEDGE"),
    "convection_diffusion": (".convection_diffusion", "KNOWLEDGE"),
    "nonlinear":            (".nonlinear",            "KNOWLEDGE"),
    "helmholtz":            (".helmholtz",            "KNOWLEDGE"),
    "eigenvalue":           (".eigenvalue",           "KNOWLEDGE"),
    "wave":                 (".wave",                 "KNOWLEDGE"),
    "hp_adaptive":          (".hp_adaptive",          "KNOWLEDGE"),
    "dg_transport":         (".dg_transport",         "KNOWLEDGE"),
    "hyperelasticity":      (".hyperelasticity",      "KNOWLEDGE"),
    "parallel_poisson":     (".parallel",             "KNOWLEDGE"),
    "_general":             (".poisson",              "GENERAL_KNOWLEDGE"),
}

# Caches
_TEMPLATE_CACHE: dict[str, Callable] = {}
_KNOWLEDGE_CACHE: dict[str, dict] = {}


def get_template(key: str) -> Callable:
    """Return the template generator function for a given key."""
    if key in _TEMPLATE_CACHE:
        return _TEMPLATE_CACHE[key]

    if key not in _TEMPLATE_SPECS:
        raise ValueError(
            f"No deal.II template for {key}. "
            f"Available: {sorted(_TEMPLATE_SPECS)}"
        )

    module_path, func_name = _TEMPLATE_SPECS[key]
    mod = importlib.import_module(module_path, package=__name__)
    func = getattr(mod, func_name)
    _TEMPLATE_CACHE[key] = func
    return func


def get_knowledge(physics: str) -> dict:
    """Return knowledge dict for a physics type."""
    if physics in _KNOWLEDGE_CACHE:
        return _KNOWLEDGE_CACHE[physics]

    if physics not in _KNOWLEDGE_SPECS:
        return {}

    module_path, dict_name = _KNOWLEDGE_SPECS[physics]
    try:
        mod = importlib.import_module(module_path, package=__name__)
        knowledge = getattr(mod, dict_name)
        _KNOWLEDGE_CACHE[physics] = knowledge
        return knowledge
    except (ImportError, AttributeError) as exc:
        logger.debug("Cannot load knowledge for %r: %s", physics, exc)
        return {}


def list_template_keys() -> list[str]:
    """Return sorted list of all available template keys."""
    return sorted(_TEMPLATE_SPECS)
