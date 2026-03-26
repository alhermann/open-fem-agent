"""Pure thermal analysis generator for 4C.

Covers standalone heat conduction without structural coupling (use TSI for coupled).
"""

from __future__ import annotations
from typing import Any
from .base import BaseGenerator


class ThermoGenerator(BaseGenerator):
    """Generator for pure thermal (heat conduction) problems in 4C."""

    module_key = "thermo"
    display_name = "Pure Thermal Analysis"
    problem_type = "Thermo"

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Standalone thermal analysis — heat conduction, convection BCs, "
                "radiation.  For thermal-structure coupling, use TSI instead."
            ),
            "yaml_section": "THERMAL DYNAMIC",
            "elements": {
                "2D": ["THERMO QUAD4", "THERMO QUAD9", "THERMO TRI3"],
                "3D": ["THERMO HEX8", "THERMO HEX27", "THERMO TET4", "THERMO TET10"],
            },
            "time_integration": {
                "Statics": "Steady-state thermal analysis",
                "OneStepTheta": "Transient with theta method (theta=1 for backward Euler)",
                "GenAlpha": "Generalized-alpha for transient thermal",
            },
            "materials": {
                "MAT_Fourier": {
                    "parameters": {
                        "CAPA": "Heat capacity [J/(m^3 K)]",
                        "CONDUCT": "Thermal conductivity [W/(m K)]",
                    },
                },
            },
            "boundary_conditions": {
                "DESIGN SURF THERMO DIRICH CONDITIONS": "Prescribed temperature",
                "DESIGN SURF THERMO NEUMANN CONDITIONS": "Prescribed heat flux",
                "DESIGN SURF THERMO CONVECTION CONDITIONS": "Convective heat transfer (h, T_inf)",
            },
            "pitfalls": [
                "PROBLEMTYPE: Thermo (not Scalar_Transport — different element types)",
                "INITIALFIELD: field_by_function with INITFUNCNO pointing to a FUNCT",
                "CAPA is volumetric heat capacity (rho*c_p), not specific heat capacity",
                "For coupled thermal-structural: use TSI, not standalone Thermo",
                "THERMO elements are separate from TRANSP (scalar transport) elements",
            ],
        }

    def list_variants(self) -> list[dict[str, str]]:
        return [{"name": "thermo_2d", "description": "2D steady-state heat conduction"},
                {"name": "thermo_3d", "description": "3D transient heat conduction"}]

    def get_template(self, variant: str = "thermo_2d") -> str:
        return "# Thermal template — use THERMO QUAD4/HEX8 elements with MAT_Fourier"

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        return []
