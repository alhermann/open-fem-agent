"""Membrane element generator for 4C.

Covers thin membrane/shell elements for structural analysis.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class MembraneGenerator(BaseGenerator):
    """Generator for membrane/thin shell problems in 4C."""

    module_key = "membrane"
    display_name = "Membrane Elements"
    problem_type = "Structure"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Thin membrane elements for inflatable structures, fabric, "
                "biological tissue.  No bending stiffness — pure in-plane stress."
            ),
            "elements": {
                "2D": ["MEMBRANE TRI3", "MEMBRANE TRI6", "MEMBRANE QUAD4", "MEMBRANE QUAD9"],
            },
            "materials": ["MAT_ElastHyper (with membrane kinematics)",
                          "MAT_Struct_StVenantKirchhoff"],
            "pitfalls": [
                "Membranes have zero bending stiffness — prone to wrinkling",
                "Use prestress or pressure loading to stabilize",
                "For wrinkling: enable wrinkling model in material definition",
                "THICK parameter defines membrane thickness",
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "membrane_2d", "description": "Membrane under pressure loading"}]

    def get_template(self, variant: str = "membrane_2d") -> str:
        return "# Membrane template — use MEMBRANE TRI3/QUAD4 elements with appropriate material"

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
