"""Nonlinear PDE templates for deal.II.

Based on deal.II tutorial step-15 (minimal surface, Newton method).
"""


def _nonlinear_minimal_surface_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    Based on deal.II step-15 (Newton method).
    """
    refinements = params.get("refinements", 4)
    return f'''\
/* Minimal surface equation (nonlinear) — Newton method — deal.II step-15 */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <fstream>
#include <cmath>
using namespace dealii;

template <int dim>
class BoundaryValues : public Function<dim> {{
public:
  double value(const Point<dim> &p, const unsigned int) const override {{
    return std::sin(2 * numbers::PI * (p[0] + p[1]));
  }}
}};

int main() {{
  const int dim = 2;
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0, 1);
  tria.refine_global({refinements});

  FE_Q<dim> fe(1);
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);

  AffineConstraints<double> constraints;
  VectorTools::interpolate_boundary_values(dof_handler, 0, BoundaryValues<dim>(), constraints);
  constraints.close();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sp;
  sp.copy_from(dsp);

  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sp);
  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> system_rhs(dof_handler.n_dofs());

  // Initial guess: interpolate boundary values
  VectorTools::interpolate(dof_handler, BoundaryValues<dim>(), solution);

  QGauss<dim> quadrature(fe.degree + 1);
  const unsigned int dpc = fe.n_dofs_per_cell();

  // Newton iterations
  for (unsigned int newton_step = 0; newton_step < 20; ++newton_step) {{
    system_matrix = 0;
    system_rhs = 0;

    FEValues<dim> fe_values(fe, quadrature,
      update_values | update_gradients | update_JxW_values);
    FullMatrix<double> cell_matrix(dpc, dpc);
    Vector<double> cell_rhs(dpc);
    std::vector<types::global_dof_index> local_dof_indices(dpc);
    std::vector<Tensor<1, dim>> old_gradients(quadrature.size());

    for (const auto &cell : dof_handler.active_cell_iterators()) {{
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs = 0;
      fe_values.get_function_gradients(solution, old_gradients);

      for (unsigned int q = 0; q < quadrature.size(); ++q) {{
        const double coeff = 1.0 / std::sqrt(1 + old_gradients[q] * old_gradients[q]);
        const double coeff3 = coeff * coeff * coeff;
        for (unsigned int i = 0; i < dpc; ++i) {{
          for (unsigned int j = 0; j < dpc; ++j)
            cell_matrix(i, j) += (coeff * fe_values.shape_grad(i, q) * fe_values.shape_grad(j, q)
                                  - coeff3 * (fe_values.shape_grad(i, q) * old_gradients[q])
                                           * (fe_values.shape_grad(j, q) * old_gradients[q]))
                                 * fe_values.JxW(q);
          cell_rhs(i) -= coeff * fe_values.shape_grad(i, q) * old_gradients[q] * fe_values.JxW(q);
        }}
      }}
      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_matrix, cell_rhs, local_dof_indices,
                                             system_matrix, system_rhs);
    }}

    SolverControl sc(1000, 1e-12);
    SolverCG<Vector<double>> solver(sc);
    PreconditionSSOR<SparseMatrix<double>> preconditioner;
    preconditioner.initialize(system_matrix, 1.2);
    Vector<double> newton_update(dof_handler.n_dofs());
    solver.solve(system_matrix, newton_update, system_rhs, preconditioner);
    constraints.distribute(newton_update);

    solution += newton_update;
    double residual = system_rhs.l2_norm();
    std::cout << "Newton step " << newton_step << ": residual = " << residual << std::endl;
    if (residual < 1e-8) break;
  }}

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, "solution");
  data_out.build_patches();
  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);
  std::cout << "Nonlinear solve complete." << std::endl;
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "Nonlinear PDEs: minimal surface (step-15), Newton (step-72 AD)",
    "tutorial_steps": ["step-15 (manual Newton)", "step-72 (AD-assisted Newton)",
                      "step-77 (SUNDIALS KINSOL)"],
    "function_space": "FE_Q<dim>(1) typically",
    "solver": "Newton-Raphson with line search. AD: Sacado/ADOL-C for tangent",
    "pitfalls": [
        "Newton needs good initial guess (interpolate boundary values)",
        "Line search prevents divergence: alpha * delta_u",
        "AssembleLinearization must update with current solution",
        "For AD: use Differentiation::AD::EnergyFunctional (step-72)",
        "SUNDIALS KINSOL can replace hand-written Newton (step-77)",
    ],
}
