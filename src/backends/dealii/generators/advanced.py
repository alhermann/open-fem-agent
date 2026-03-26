"""deal.II advanced physics generators and knowledge.

Covers: mixed_laplacian, compressible_euler, time_dependent, matrix_free,
multigrid, multiphysics, obstacle, topology_opt, error_estimation, phase_field.
"""


def _placeholder_dealii(name: str, steps: str, desc: str) -> str:
    """Generate a placeholder C++ file referencing the deal.II tutorial."""
    return f'''\
/* {desc} — deal.II
 * Reference tutorials: {steps}
 * See: https://www.dealii.org/current/doxygen/deal.II/{steps.split("(")[0].strip().replace(" ", "_")}.html
 */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <iostream>
using namespace dealii;
int main() {{
    std::cout << "{name} solver — reference: {steps}" << std::endl;
    std::cout << "See deal.II tutorial for full implementation" << std::endl;
    return 0;
}}
'''


def _mixed_laplacian_2d(p: dict) -> str:
    return _placeholder_dealii("Mixed Laplacian", "step-20 (Raviart-Thomas)",
        "Mixed formulation with H(div) elements")

def _compressible_euler_2d(p: dict) -> str:
    return _placeholder_dealii("Compressible Euler", "step-33 (conservation laws), step-69 (Euler)",
        "Compressible gas dynamics with shock capturing")

def _time_dependent_heat_2d(p: dict) -> str:
    return _placeholder_dealii("Time-dependent heat", "step-26 (adaptive heat eq)",
        "Transient heat equation with adaptive mesh refinement")

def _time_dependent_wave_2d(p: dict) -> str:
    return _placeholder_dealii("Time-dependent wave", "step-23 (wave eq), step-48 (parallel)",
        "Second-order wave equation with Newmark time stepping")

def _time_dependent_ns_2d(p: dict) -> str:
    return _placeholder_dealii("Time-dependent NS", "step-35 (Boussinesq)",
        "Transient buoyancy-driven flow with Boussinesq approximation")

def _matrix_free_2d(p: dict) -> str:
    return _placeholder_dealii("Matrix-free", "step-37, step-59",
        "Matrix-free operator evaluation for high-performance FEM")

def _multigrid_2d(p: dict) -> str:
    return _placeholder_dealii("Multigrid", "step-16 (GMG), step-50 (parallel GMG)",
        "Geometric multigrid preconditioner for iterative solvers")

def _multiphysics_2d(p: dict) -> str:
    return _placeholder_dealii("Multiphysics", "step-21 (two-phase), step-43 (two-phase NS)",
        "Two-phase flow with Darcy or Navier-Stokes")

def _obstacle_2d(p: dict) -> str:
    return _placeholder_dealii("Obstacle/contact", "step-41",
        "Variational inequality / obstacle problem (contact)")

def _topology_opt_2d(p: dict) -> str:
    return _placeholder_dealii("Topology optimization", "step-79 (SIMP)",
        "SIMP topology optimization for compliance minimization")

def _error_estimation_2d(p: dict) -> str:
    return _placeholder_dealii("Error estimation", "step-14 (DWR), step-74 (refinement)",
        "Dual-weighted residual error estimation and adaptive refinement")

def _phase_field_2d(p: dict) -> str:
    return _placeholder_dealii("Phase field", "step-63",
        "Phase-field / advection-diffusion-reaction with SUPG stabilization")

def _dg_advection_2d(p: dict) -> str:
    return _placeholder_dealii("DG advection-reaction", "step-12 (DG), step-39 (DG+MG)",
        "Discontinuous Galerkin for advection with upwind flux")

def _cg_dg_coupled_2d(p: dict) -> str:
    return _placeholder_dealii("CG-DG coupled", "step-46",
        "Mixed continuous-discontinuous Galerkin methods")

def _optimal_control_2d(p: dict) -> str:
    return _placeholder_dealii("Optimal control / AD", "step-72 (automatic differentiation)",
        "Automatic differentiation for tangent assembly and optimization")


KNOWLEDGE = {
    "mixed_laplacian": {
        "description": "Mixed Laplacian with Raviart-Thomas H(div) elements (step-20)",
        "function_space": "FE_RaviartThomas + FE_DGQ for flux-pressure formulation",
        "pitfalls": ["H(div) elements — different DOF structure from H1",
                     "Schur complement solver for saddle-point system"],
    },
    "compressible_euler": {
        "description": "Compressible Euler equations — shock-capturing DG (step-33, step-69)",
        "methods": ["Lax-Friedrichs flux", "HLLC Riemann solver", "entropy viscosity"],
        "pitfalls": ["Shock capturing requires artificial viscosity or limiting",
                     "CFL condition for explicit time stepping",
                     "Mach number scaling affects conditioning"],
    },
    "time_dependent_heat": {
        "description": "Transient heat equation with AMR (step-26)",
        "time_integration": ["backward Euler", "Crank-Nicolson", "BDF2"],
        "pitfalls": ["Adaptive mesh refinement requires solution transfer between meshes",
                     "CFL for explicit; unconditionally stable for implicit"],
    },
    "time_dependent_wave": {
        "description": "Second-order wave equation (step-23, step-48)",
        "time_integration": ["Newmark-beta", "leapfrog"],
        "pitfalls": ["Energy conservation — use symplectic integrators",
                     "CFL: dt < h/c for explicit schemes"],
    },
    "time_dependent_ns": {
        "description": "Transient Boussinesq flow — buoyancy-driven convection (step-35)",
        "pitfalls": ["Rayleigh number controls flow regime",
                     "Requires NS + energy equation coupling"],
    },
    "matrix_free": {
        "description": "Matrix-free operator evaluation — high performance FEM (step-37, step-59)",
        "performance": "10-100x faster than sparse matrix for high-order elements",
        "pitfalls": ["Requires tensor-product elements (FE_Q, FE_DGQ)",
                     "No matrix assembly — operator is applied on-the-fly",
                     "Geometric multigrid essential for preconditioning"],
    },
    "multigrid": {
        "description": "Geometric multigrid preconditioner (step-16, step-50)",
        "types": ["h-multigrid (mesh hierarchy)", "p-multigrid (polynomial degree)"],
        "pitfalls": ["Smoother choice: Chebyshev for SPD, GMRES for indefinite",
                     "Coarse grid solver: direct (Amesos) or iterative"],
    },
    "multiphysics_dealii": {
        "description": "Two-phase flow and multi-physics coupling (step-21, step-43)",
        "pitfalls": ["Darcy flow (step-21) vs full NS (step-43)",
                     "Interface tracking: level-set or phase-field"],
    },
    "obstacle_problem": {
        "description": "Variational inequality / contact / obstacle problem (step-41)",
        "method": "Active set strategy — project onto feasible set each Newton step",
        "pitfalls": ["Non-smooth problem — requires special solver (active set, penalty)"],
    },
    "topology_opt_dealii": {
        "description": "SIMP topology optimization (step-79)",
        "method": "SIMP with density filtering and MMA optimizer",
        "pitfalls": ["Penalization factor p=3", "Filter radius prevents checkerboard",
                     "Mesh-dependent without proper regularization"],
    },
    "error_estimation": {
        "description": "Dual-weighted residual (DWR) error estimation (step-14, step-74)",
        "method": "Solve dual/adjoint problem, weight residual for goal-oriented refinement",
        "pitfalls": ["Dual problem requires adjoint assembly",
                     "Higher-order dual solution needed for effectivity index"],
    },
    "phase_field": {
        "description": "Phase-field / advection-diffusion-reaction with SUPG (step-63)",
        "pitfalls": ["SUPG stabilization for advection-dominated problems",
                     "Peclet number determines stabilization strength"],
    },
    "dg_advection_reaction": {
        "description": "DG for advection with upwind flux (step-12, step-39)",
        "pitfalls": ["Upwind flux for stability", "DG + multigrid in step-39"],
    },
    "cg_dg_coupled": {
        "description": "Mixed CG-DG methods (step-46)",
        "pitfalls": ["Different FE spaces in different subdomains",
                     "Interface conditions between CG and DG regions"],
    },
    "optimal_control": {
        "description": "Automatic differentiation for tangent/residual (step-72)",
        "method": "Sacado AD for automatic tangent assembly",
        "pitfalls": ["AD adds overhead but eliminates hand-coded tangent errors",
                     "Requires Trilinos with Sacado support"],
    },
}

GENERATORS = {
    "mixed_laplacian_2d": _mixed_laplacian_2d,
    "compressible_euler_2d": _compressible_euler_2d,
    "time_dependent_heat_2d": _time_dependent_heat_2d,
    "time_dependent_wave_2d": _time_dependent_wave_2d,
    "time_dependent_ns_2d": _time_dependent_ns_2d,
    "matrix_free_2d": _matrix_free_2d,
    "multigrid_2d": _multigrid_2d,
    "multiphysics_dealii_2d": _multiphysics_2d,
    "obstacle_problem_2d": _obstacle_2d,
    "topology_opt_dealii_2d": _topology_opt_2d,
    "error_estimation_2d": _error_estimation_2d,
    "phase_field_2d": _phase_field_2d,
    "dg_advection_reaction_2d": _dg_advection_2d,
    "cg_dg_coupled_2d": _cg_dg_coupled_2d,
    "optimal_control_2d": _optimal_control_2d,
}
