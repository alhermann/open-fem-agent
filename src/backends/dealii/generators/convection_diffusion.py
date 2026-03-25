"""Convection-diffusion templates for deal.II.

Based on deal.II tutorial step-9.
"""


def _convdiff_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    Based on deal.II step-9.
    """
    refinements = params.get("refinements", 5)
    eps = params.get("diffusion", 0.01)
    return f'''\
/* Convection-diffusion: SUPG stabilized — deal.II (step-9 inspired) */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <fstream>
#include <cmath>
using namespace dealii;

int main() {{
  const int dim = 2;
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0, 1);
  tria.refine_global({refinements});

  FE_Q<dim> fe(1);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sp;
  sp.copy_from(dsp);

  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sp);
  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> system_rhs(dof_handler.n_dofs());

  const double eps = {eps};
  const Tensor<1, dim> beta({{{{1.0, 0.5}}}});

  QGauss<dim> quadrature(2);
  FEValues<dim> fe_values(fe, quadrature,
    update_values | update_gradients | update_JxW_values | update_quadrature_points);

  const unsigned int dpc = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dpc, dpc);
  Vector<double> cell_rhs(dpc);
  std::vector<types::global_dof_index> local_dof_indices(dpc);

  for (const auto &cell : dof_handler.active_cell_iterators()) {{
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_rhs = 0;
    const double h = cell->diameter();
    const double Pe = beta.norm() * h / (2.0 * eps);
    const double tau = (Pe > 1.0) ? h / (2.0 * beta.norm()) * (1.0 - 1.0 / Pe) : 0.0;

    for (unsigned int q = 0; q < quadrature.size(); ++q) {{
      for (unsigned int i = 0; i < dpc; ++i) {{
        const double phi_i = fe_values.shape_value(i, q);
        const Tensor<1, dim> grad_phi_i = fe_values.shape_grad(i, q);
        const double supg_test = phi_i + tau * (beta * grad_phi_i);
        for (unsigned int j = 0; j < dpc; ++j) {{
          const Tensor<1, dim> grad_phi_j = fe_values.shape_grad(j, q);
          const double advection = beta * grad_phi_j;
          cell_matrix(i, j) += (eps * grad_phi_i * grad_phi_j + supg_test * advection)
                               * fe_values.JxW(q);
        }}
        cell_rhs(i) += 1.0 * supg_test * fe_values.JxW(q);
      }}
    }}
    cell->get_dof_indices(local_dof_indices);
    for (unsigned int i = 0; i < dpc; ++i) {{
      for (unsigned int j = 0; j < dpc; ++j)
        system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
      system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }}
  }}

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0,
    Functions::ZeroFunction<dim>(), boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

  SolverControl sc(1000, 1e-10);
  SolverGMRES<Vector<double>> solver(sc);
  PreconditionSSOR<SparseMatrix<double>> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);
  solver.solve(system_matrix, solution, system_rhs, preconditioner);

  std::cout << "ConvDiff: " << dof_handler.n_dofs() << " DOFs, "
            << sc.last_step() << " GMRES iters" << std::endl;
  std::cout << "max(u) = " << solution.linfty_norm() << std::endl;

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "Convection-diffusion: SUPG (step-9), DG (step-12), HDG (step-51)",
    "tutorial_steps": ["step-9 (SUPG streamline diffusion)", "step-12 (DG upwind)",
                      "step-30 (anisotropic refinement)", "step-51 (HDG)",
                      "step-63 (GMG with block smoothers)"],
    "function_space": "FE_Q(1) for SUPG, FE_DGQ(p) for DG",
    "solver": "BiCGStab + Jacobi for SUPG; direct for DG (block-diagonal)",
    "pitfalls": [
        "SUPG: tau = h/(2|b|) * (coth(Pe) - 1/Pe), Pe = |b|*h/(2*eps)",
        "DG: need FEInterfaceValues for jump/average on faces",
        "Upwind flux: use the value from the element where b·n > 0",
        "For high Peclet: either reduce h or use DG (SUPG may oscillate)",
    ],
}
