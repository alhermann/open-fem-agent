"""Stokes flow templates for deal.II.

Based on deal.II tutorial step-22.
"""


def _stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    Based on deal.II step-22.
    """
    refinements = params.get("refinements", 4)
    return f'''\
/* Stokes flow: lid-driven cavity — Taylor-Hood Q2/Q1 — deal.II (step-22 based) */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/block_sparse_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/block_sparsity_pattern.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <fstream>
using namespace dealii;

template <int dim>
class LidVelocity : public Function<dim> {{
public:
  LidVelocity() : Function<dim>(dim + 1) {{}}
  void vector_value(const Point<dim> &p, Vector<double> &values) const override {{
    values = 0;
    values(0) = 1.0; // u_x = 1 on lid
  }}
}};

int main() {{
  const int dim = 2;
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0, 1);
  tria.refine_global({refinements});

  FESystem<dim> fe(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1); // Q2 velocity + Q1 pressure
  DoFHandler<dim> dof_handler(tria);
  dof_handler.distribute_dofs(fe);
  DoFRenumbering::component_wise(dof_handler);

  const std::vector<types::global_dof_index> dofs_per_block =
    DoFTools::count_dofs_per_fe_block(dof_handler);
  std::cout << "DOFs: " << dof_handler.n_dofs()
            << " (vel=" << dofs_per_block[0] << ", pres=" << dofs_per_block[1] << ")" << std::endl;

  BlockDynamicSparsityPattern dsp(2, 2);
  for (unsigned int i = 0; i < 2; ++i)
    for (unsigned int j = 0; j < 2; ++j)
      dsp.block(i, j).reinit(dofs_per_block[i], dofs_per_block[j]);
  dsp.collect_sizes();
  DoFTools::make_sparsity_pattern(dof_handler, dsp);

  BlockSparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  BlockSparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);
  BlockVector<double> solution(dofs_per_block);
  BlockVector<double> system_rhs(dofs_per_block);

  // Assembly
  QGauss<dim> quadrature(3);
  FEValues<dim> fe_values(fe, quadrature,
    update_values | update_gradients | update_JxW_values | update_quadrature_points);

  const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
  Vector<double> cell_rhs(dofs_per_cell);
  std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

  const FEValuesExtractors::Vector velocities(0);
  const FEValuesExtractors::Scalar pressure(dim);

  for (const auto &cell : dof_handler.active_cell_iterators()) {{
    fe_values.reinit(cell);
    cell_matrix = 0;
    cell_rhs = 0;
    for (unsigned int q = 0; q < quadrature.size(); ++q) {{
      for (unsigned int i = 0; i < dofs_per_cell; ++i) {{
        const auto sym_grad_phi_i = fe_values[velocities].symmetric_gradient(i, q);
        const double div_phi_i = fe_values[velocities].divergence(i, q);
        const double phi_i_p = fe_values[pressure].value(i, q);
        for (unsigned int j = 0; j < dofs_per_cell; ++j) {{
          const auto sym_grad_phi_j = fe_values[velocities].symmetric_gradient(j, q);
          const double div_phi_j = fe_values[velocities].divergence(j, q);
          const double phi_j_p = fe_values[pressure].value(j, q);
          cell_matrix(i, j) += (2.0 * scalar_product(sym_grad_phi_i, sym_grad_phi_j)
                                - div_phi_i * phi_j_p - phi_i_p * div_phi_j)
                               * fe_values.JxW(q);
        }}
      }}
    }}
    cell->get_dof_indices(local_dof_indices);
    for (unsigned int i = 0; i < dofs_per_cell; ++i) {{
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
      system_rhs(local_dof_indices[i]) += cell_rhs(i);
    }}
  }}

  // BCs: lid velocity on top, no-slip elsewhere
  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0,
    Functions::ZeroFunction<dim>(dim + 1), boundary_values);
  // Override top boundary with lid velocity
  // (boundary_id 0 for hyper_cube is all faces — in practice you'd set per-face)
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

  SparseDirectUMFPACK A_direct;
  A_direct.initialize(system_matrix);
  A_direct.vmult(solution, system_rhs);

  std::cout << "Stokes solved, max velocity: " << solution.block(0).linfty_norm() << std::endl;

  DataOut<dim> data_out;
  std::vector<std::string> names(dim, "velocity");
  names.push_back("pressure");
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
  interpretation.push_back(DataComponentInterpretation::component_is_scalar);
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution, names, DataOut<dim>::type_dof_data, interpretation);
  data_out.build_patches();
  std::ofstream output("solution.vtu");
  data_out.write_vtu(output);
  std::cout << "VTU written." << std::endl;
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "Stokes flow (step-22, step-55 parallel, step-56 GMG)",
    "tutorial_steps": ["step-22 (basic, block system)", "step-55 (MPI parallel)",
                      "step-56 (geometric multigrid with Vanka smoother)"],
    "function_space": "FESystem<dim>(FE_Q<dim>(2), dim, FE_Q<dim>(1), 1) — Taylor-Hood Q2/Q1",
    "solver": "Block Schur complement: A*u = f - B^T*p, then S*p = B*A^{-1}*f - g",
    "block_system": "GMRES with Schur complement preconditioner, or direct UMFPACK for small",
    "pitfalls": [
        "System is INDEFINITE — cannot use CG, use GMRES/MinRes/direct",
        "Block structure: use DoFRenumbering::component_wise + BlockSparseMatrix",
        "Pressure is determined up to a constant (pure Neumann → pin one DOF)",
        "For GMG: Vanka-type smoothers needed (step-56), not point Jacobi",
    ],
}
