"""Mixture/multiscale material generator for 4C.

Covers fiber-reinforced composites, biological tissue mixtures.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class MixtureGenerator(BaseGenerator):
    """Generator for mixture/composite material problems in 4C."""

    module_key = "mixture"
    display_name = "Mixture/Composite Materials"
    problem_type = "Structure"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Mixture theory for fiber-reinforced composites and biological tissues.  "
                "Multiple material constituents with individual constitutive laws "
                "combined via mixture rules."
            ),
            "materials": {
                "MAT_Mixture": "General mixture material with multiple constituents",
                "constituents": [
                    "MAT_ElastHyper (isotropic ground substance)",
                    "MAT_Muscle_Weickenmeier (skeletal muscle)",
                    "MAT_Muscle_Giantesio (active muscle)",
                    "Fiber families with anisotropic response",
                ],
            },
            "applications": ["arterial wall mechanics", "tendon/ligament",
                             "muscle tissue", "fiber-reinforced polymers",
                             "growth and remodeling"],
            "pitfalls": [
                "Mixture rule: stress = sum(volume_fraction_i * stress_i)",
                "Fiber direction must be specified per element or via field",
                "Growth and remodeling requires time-dependent mass sources",
                "Incompressibility constraint handled via penalty or mixed formulation",
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "mixture_3d", "description": "Mixture material under loading"}]

    def get_template(self, variant: str = "mixture_3d") -> str:
        return "# Mixture template — use MAT_Mixture with multiple constituents"

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
