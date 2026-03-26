"""0-D cardiovascular / windkessel models generator for 4C.

Covers lumped-parameter heart models, windkessel afterload, closed-loop circulation.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class Cardiovascular0DGenerator(BaseGenerator):
    """Generator for 0-D cardiovascular models in 4C."""

    module_key = "cardiovascular0d"
    display_name = "0-D Cardiovascular (Windkessel)"
    problem_type = "Structure"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Lumped-parameter (0-D) cardiovascular models: windkessel afterload, "
                "closed-loop circulation, time-varying elastance heart model.  "
                "Coupled to 3-D fluid or structure via surface conditions."
            ),
            "models": {
                "windkessel_3element": "R_p, C, R_d — proximal resistance, compliance, distal resistance",
                "windkessel_4element": "R_p, L, C, R_d — adds inductance",
                "heart_time_varying_elastance": "E(t) model with active contraction",
                "closed_loop": "Full circulation: heart + arterial + venous + pulmonary",
            },
            "coupling": [
                "DESIGN SURF CARDIOVASCULAR0D CONDITIONS",
                "Coupled to 3-D fluid outflow or structural cavity volume",
            ],
            "applications": ["cardiac simulation", "hemodynamics", "afterload modeling",
                             "valve simulation", "ventricular assist device"],
            "pitfalls": [
                "Windkessel parameters must match the vascular impedance",
                "Time-varying elastance requires cardiac cycle timing parameters",
                "Coupling to 3-D: cavity volume computed from surface integral",
                "Initial conditions: set initial pressures in the 0-D model",
                "Typically used with fluid or FSI, not standalone",
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "windkessel_3d", "description": "3-element windkessel coupled to 3D"}]

    def get_template(self, variant: str = "windkessel_3d") -> str:
        return "# Cardiovascular0D template — use DESIGN SURF CARDIOVASCULAR0D CONDITIONS"

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
