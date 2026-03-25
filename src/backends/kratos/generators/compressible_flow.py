"""Kratos compressible flow generators and knowledge.

Covers compressible potential flow and compressible Euler/Navier-Stokes.
Applications: CompressiblePotentialFlowApplication.
"""


def _compressible_potential_2d(params: dict) -> str:
    """FORMAT TEMPLATE — Compressible potential flow around airfoil."""
    return '''\
"""Compressible potential flow — Kratos CompressiblePotentialFlowApplication"""
import json
try:
    import KratosMultiphysics as KM
    import KratosMultiphysics.CompressiblePotentialFlowApplication
    print("CompressiblePotentialFlowApplication available")
    summary = {"note": "CompressiblePotentialFlowApplication available",
               "capabilities": ["subsonic_potential", "transonic_potential", "full_potential"]}
except ImportError:
    print("CompressiblePotentialFlowApplication not installed")
    print("Install: pip install KratosCompressiblePotentialFlowApplication")
    summary = {"note": "not installed"}
with open("results_summary.json", "w") as f: json.dump(summary, f, indent=2)
'''


KNOWLEDGE = {
    "compressible_potential": {
        "description": "Compressible potential flow (subsonic/transonic) around aerodynamic bodies",
        "application": "CompressiblePotentialFlowApplication",
        "elements": {
            "2D": ["IncompressiblePotentialFlowElement2D3N", "CompressiblePotentialFlowElement2D3N",
                   "TransonicPerturbationPotentialFlowElement2D3N"],
            "3D": ["IncompressiblePotentialFlowElement3D4N", "CompressiblePotentialFlowElement3D4N"],
        },
        "solver_types": ["potential_flow_solver (linear/nonlinear)"],
        "pitfalls": [
            "Far-field BC: use PotentialWallCondition for solid walls",
            "Freestream: set FREESTREAM_VELOCITY and MACH_INFINITY",
            "Transonic: requires shock-capturing stabilization",
            "Lift/drag computed from pressure integration on body surface",
        ],
    },
}

GENERATORS = {
    "compressible_potential_2d": _compressible_potential_2d,
}
