"""Shell element generator for 4C.

Covers Kirchhoff-Love and Reissner-Mindlin shell elements.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class ShellGenerator(BaseGenerator):
    """Generator for shell structure problems in 4C."""

    module_key = "shell"
    display_name = "Shell Elements"
    problem_type = "Structure"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Thin and thick shell elements for plates, curved shells, "
                "and general 3D surface structures.  Includes Kirchhoff-Love "
                "(thin, no transverse shear) and Reissner-Mindlin (thick, "
                "transverse shear) formulations."
            ),
            "elements": {
                "Kirchhoff-Love": ["SHELL KIRCHHOFF TRI3", "SHELL KIRCHHOFF QUAD4",
                                   "SHELL KIRCHHOFF QUAD9"],
                "Reissner-Mindlin": ["SHELL REISSNER TRI3", "SHELL REISSNER QUAD4",
                                     "SHELL REISSNER QUAD9"],
                "solid-shell": ["SOLIDSHELL HEX8 (continuum shell with 3D topology)"],
            },
            "materials": ["MAT_Struct_StVenantKirchhoff", "MAT_ElastHyper",
                          "MAT_Struct_MicroMaterial (multiscale)"],
            "pitfalls": [
                "Kirchhoff shells need C1 continuity — use NURBS or DKT formulation",
                "Reissner-Mindlin shells can lock for thin shells — use reduced integration",
                "THICK parameter is the shell thickness",
                "Director vector must be specified or auto-computed from element normal",
                "Shell elements produce both in-plane forces and bending moments",
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "shell_3d", "description": "Shell structure under loading"}]

    def get_template(self, variant: str = "shell_3d") -> str:
        return "# Shell template — use SHELL REISSNER QUAD4 for general purpose"

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
