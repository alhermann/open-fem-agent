"""Kratos PFEM (Particle Finite Element Method) generators and knowledge.

Covers free-surface flows, sloshing, fluid-structure with topology changes.
Applications: PfemFluidDynamicsApplication, PfemApplication, PFEM2Application.
"""


def _pfem_fluid_2d(params: dict) -> str:
    """FORMAT TEMPLATE — PFEM free-surface fluid simulation."""
    return '''\
"""PFEM free-surface flow — Kratos PfemFluidDynamicsApplication"""
import json
try:
    import KratosMultiphysics as KM
    import KratosMultiphysics.PfemFluidDynamicsApplication
    print("PfemFluidDynamicsApplication available")
    summary = {"note": "PfemFluidDynamicsApplication available",
               "capabilities": ["free_surface", "sloshing", "dam_break", "wave_breaking",
                                "fluid_structure_topology_change"]}
except ImportError:
    print("PfemFluidDynamicsApplication not installed")
    print("Install: pip install KratosPfemFluidDynamicsApplication")
    summary = {"note": "not installed"}
with open("results_summary.json", "w") as f: json.dump(summary, f, indent=2)
'''


KNOWLEDGE = {
    "pfem_fluid": {
        "description": "Particle FEM for free-surface flows (dam break, sloshing, wave impact)",
        "application": "PfemFluidDynamicsApplication (pip install KratosPfemFluidDynamicsApplication)",
        "elements": {
            "2D": ["TwoStepUpdatedLagrangianVPImplicitNodallyIntegratedElement2D3N",
                   "TwoStepUpdatedLagrangianVPImplicitFluidElement2D3N"],
            "3D": ["TwoStepUpdatedLagrangianVPImplicitNodallyIntegratedElement3D4N"],
        },
        "capabilities": ["free_surface_tracking", "remeshing", "alpha_shape_boundary_detection",
                         "fluid_structure_with_topology_changes"],
        "solver_types": ["two_step_v_p_solver (velocity-pressure split)"],
        "pitfalls": [
            "Requires DelaunayMeshingApplication for remeshing",
            "Alpha-shape parameter controls free-surface detection (default ~1.25)",
            "Time step must be small enough for remeshing stability",
            "Output: particles move, so mesh changes every step",
        ],
    },
    "pfem_solid": {
        "description": "PFEM for large-deformation solid mechanics with remeshing",
        "application": "PfemSolidMechanicsApplication",
        "capabilities": ["large_deformation_solids", "cutting", "forming", "erosion"],
    },
    "pfem2": {
        "description": "PFEM2 (streamline integration) for two-phase flows",
        "application": "PFEM2Application",
        "capabilities": ["two_phase_flow", "interface_tracking", "bubble_dynamics"],
    },
}

GENERATORS = {
    "pfem_fluid_2d": _pfem_fluid_2d,
}
