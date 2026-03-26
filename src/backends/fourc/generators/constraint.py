"""Constraint/coupling condition generator for 4C.

Covers multi-point constraints, rigid body constraints, periodic BCs.
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class ConstraintGenerator(BaseGenerator):
    """Generator for constraint problems in 4C."""

    module_key = "constraint"
    display_name = "Constraints and Coupling"
    problem_type = "Structure"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Multi-point constraints, rigid body coupling, periodic boundary "
                "conditions, and general coupling of DOFs across discretizations."
            ),
            "condition_types": {
                "MPC": "Multi-point constraints (linear combinations of DOFs)",
                "Rigid body": "Rigid coupling of a set of nodes to a master node",
                "Periodic": "Periodic boundary conditions for unit cell analysis",
                "Mortar": "Mortar-based surface coupling (non-matching meshes)",
                "Penalty": "Penalty-based constraint enforcement",
            },
            "yaml_sections": [
                "DESIGN LINE COUPLING CONDITIONS",
                "DESIGN SURF COUPLING CONDITIONS",
                "DESIGN LINE MPC NORMAL COMPONENT CONDITIONS",
            ],
            "pitfalls": [
                "Constraint equations must be linearly independent",
                "Penalty parameter affects both accuracy and conditioning",
                "Mortar coupling requires integration on the interface",
                "Periodic BCs: master and slave surfaces must match geometrically",
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "constraint_3d", "description": "Multi-point constraint problem"}]

    def get_template(self, variant: str = "constraint_3d") -> str:
        return "# Constraint template — use DESIGN LINE/SURF COUPLING CONDITIONS"

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
