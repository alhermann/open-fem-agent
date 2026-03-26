"""deal.II Navier-Stokes generators and knowledge.

Based on step-57 (stationary NS), step-35 (Boussinesq), step-55 (Stokes MPI).
"""


def _navier_stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE — Stationary Navier-Stokes (Newton iteration)."""
    return '''\
/* Stationary Navier-Stokes — deal.II (based on step-57)
 * Newton iteration for nonlinear convective term.
 * Taylor-Hood Q2/Q1 elements. */
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <fstream>
#include <iostream>

using namespace dealii;

// Placeholder: implement Newton iteration for NS
// Reference: step-57 in deal.II tutorials
int main() {
    std::cout << "Navier-Stokes solver — see step-57 for full implementation" << std::endl;
    return 0;
}
'''


KNOWLEDGE = {
    "description": "Navier-Stokes (stationary and transient) — step-57, step-35, step-55",
    "tutorial_steps": ["step-57 (stationary NS, Newton)", "step-35 (Boussinesq buoyancy)",
                       "step-55 (Stokes, MPI parallel)"],
    "function_space": "FESystem<dim>(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1) — Taylor-Hood Q2/Q1",
    "solver": "Newton iteration for nonlinear convection term, UMFPACK or GMRES+ILU for linear sub-problems",
    "pitfalls": [
        "NS is nonlinear — requires Newton iteration or Picard/Oseen linearization",
        "Taylor-Hood Q2/Q1 satisfies inf-sup — Q1/Q1 does not (needs stabilization)",
        "Reynolds number affects convergence — start with low Re, increase gradually",
        "For time-dependent: use BDF2 or Crank-Nicolson, step-35 for Boussinesq",
        "Pressure is determined up to a constant — pin at one point or use mean-free",
    ],
}

GENERATORS = {
    "navier_stokes_2d": _navier_stokes_2d,
}
