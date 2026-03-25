"""Kratos fluid dynamics generators and knowledge."""


def _fluid_cavity_kratos(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Navier-Stokes via Kratos FluidDynamicsApplication."""
    return f'''\
"""Navier-Stokes — Kratos FluidDynamicsApplication"""
import json
try:
    import KratosMultiphysics as KM
    import KratosMultiphysics.FluidDynamicsApplication as FDA
    print("FluidDynamicsApplication available")
    # Full Kratos fluid analysis uses:
    # from KratosMultiphysics.FluidDynamicsApplication.fluid_dynamics_analysis import FluidDynamicsAnalysis
    # Requires: ProjectParameters.json + FluidMaterials.json + mesh.mdpa
    summary = {{"note": "Kratos FDA available — use FluidDynamicsAnalysis for full simulation"}}
except ImportError:
    print("FluidDynamicsApplication not installed")
    print("Install: pip install KratosFluidDynamicsApplication")
    summary = {{"note": "KratosFluidDynamicsApplication not installed"}}
with open("results_summary.json", "w") as _f: json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "fluid": {
        "description": "Incompressible Navier-Stokes via FluidDynamicsApplication (FDA)",
        "application": "FluidDynamicsApplication (pip install KratosFluidDynamicsApplication)",
        "elements": {
            "stabilized": ["VMS2D3N/3D4N (Variational Multiscale)",
                          "QSVMS2D3N/3D4N (Quasi-static VMS, default)"],
            "fractional_step": ["FractionalStep2D3N/3D4N"],
            "two_fluid": ["TwoFluidNavierStokes2D3N/3D4N (level-set free surface)"],
        },
        "solver_types": ["monolithic (navier_stokes_solver_vmsmonolithic)",
                        "fractional_step (navier_stokes_solver_fractionalstep)"],
        "stabilization": {
            "ASGS": "oss_switch=0 (Algebraic SubGrid Scale)",
            "OSS": "oss_switch=1 (Orthogonal SubScale)",
            "dynamic_tau": "Automatic stabilization parameter",
        },
        "turbulence": "k-epsilon, k-omega SST via RANSApplication",
        "pitfalls": [
            "VELOCITY (vector) and PRESSURE (scalar) are the primary variables",
            "Materials in FluidMaterials.json: DENSITY, DYNAMIC_VISCOSITY",
            "Wall BCs: no-slip (VELOCITY=0), Navier slip, wall law",
            "Inlet: impose VELOCITY vector, Outlet: impose PRESSURE=0",
            "For free surface: add DISTANCE variable (level-set signed distance)",
        ],
    },
}

GENERATORS = {
    "fluid_2d_cavity": _fluid_cavity_kratos,
}
