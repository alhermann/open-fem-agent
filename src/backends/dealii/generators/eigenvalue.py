"""Eigenvalue problem templates for deal.II.

Based on deal.II tutorial step-36 (SLEPc eigenvalue solver).
"""


def _eigenvalue_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable script. All parameter defaults are placeholders. The user/agent must set values appropriate to the specific problem being solved."""
    refinements = params.get("refinements", 5)
    n_eigenvalues = params.get("n_eigenvalues", 5)
    return f'''\
/* Eigenvalue problem: Laplacian on unit square — deal.II (step-36 inspired)
 * Find lambda, u such that: -laplacian(u) = lambda * u
 * Uses SLEPc for the generalized eigenvalue problem A*x = lambda*M*x.
 *
 * NOTE: Requires deal.II compiled with PETSc + SLEPc support.
 * Compile with: cmake -DDEAL_II_WITH_PETSC=ON -DDEAL_II_WITH_SLEPC=ON
 */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/slepc_solver.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/utilities.h>
#include <deal.II/base/index_set.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/affine_constraints.h>
#include <fstream>
#include <iostream>

using namespace dealii;

int main(int argc, char **argv)
{{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  const int dim = 2;
  Triangulation<dim> triangulation;
  GridGenerator::hyper_cube(triangulation, 0, 1);
  triangulation.refine_global({refinements});

  FE_Q<dim> fe(1);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  const unsigned int n_dofs = dof_handler.n_dofs();
  std::cout << "Eigenvalue DOFs: " << n_dofs << std::endl;

  // Constraints: homogeneous Dirichlet on all boundaries
  AffineConstraints<double> constraints;
  VectorTools::interpolate_boundary_values(dof_handler, 0,
    Functions::ZeroFunction<dim>(), constraints);
  constraints.close();

  // Sparsity
  DynamicSparsityPattern dsp(n_dofs);
  DoFTools::make_sparsity_pattern(dof_handler, dsp, constraints, false);

  // PETSc matrices
  IndexSet locally_owned(n_dofs);
  locally_owned.add_range(0, n_dofs);

  PETScWrappers::SparseMatrix stiffness_matrix;
  stiffness_matrix.reinit(locally_owned, locally_owned, dsp, MPI_COMM_WORLD);
  PETScWrappers::SparseMatrix mass_matrix;
  mass_matrix.reinit(locally_owned, locally_owned, dsp, MPI_COMM_WORLD);

  // Assemble stiffness and mass matrices
  QGauss<dim> quadrature(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature,
    update_values | update_gradients | update_JxW_values);

  const unsigned int dpc = fe.n_dofs_per_cell();
  FullMatrix<double> cell_stiffness(dpc, dpc);
  FullMatrix<double> cell_mass(dpc, dpc);
  std::vector<types::global_dof_index> local_dof_indices(dpc);

  for (const auto &cell : dof_handler.active_cell_iterators())
    {{
      fe_values.reinit(cell);
      cell_stiffness = 0;
      cell_mass = 0;

      for (unsigned int q = 0; q < quadrature.size(); ++q)
        for (unsigned int i = 0; i < dpc; ++i)
          for (unsigned int j = 0; j < dpc; ++j)
            {{
              cell_stiffness(i, j) += fe_values.shape_grad(i, q) *
                                      fe_values.shape_grad(j, q) *
                                      fe_values.JxW(q);
              cell_mass(i, j) += fe_values.shape_value(i, q) *
                                 fe_values.shape_value(j, q) *
                                 fe_values.JxW(q);
            }}

      cell->get_dof_indices(local_dof_indices);
      constraints.distribute_local_to_global(cell_stiffness, local_dof_indices, stiffness_matrix);
      constraints.distribute_local_to_global(cell_mass, local_dof_indices, mass_matrix);
    }}

  stiffness_matrix.compress(VectorOperation::add);
  mass_matrix.compress(VectorOperation::add);

  // Solve eigenvalue problem with SLEPc
  const unsigned int n_eigenvalues = {n_eigenvalues};
  std::vector<PETScWrappers::MPI::Vector> eigenvectors(n_eigenvalues);
  for (auto &v : eigenvectors)
    v.reinit(locally_owned, MPI_COMM_WORLD);
  std::vector<double> eigenvalues(n_eigenvalues);

  SLEPcWrappers::SolverKrylovSchur eigensolver(SolverControl(5000, 1e-10));
  eigensolver.set_which_eigenpairs(EigenvalueAlgorithmData::smallest_magnitude);
  eigensolver.solve(stiffness_matrix, mass_matrix, eigenvalues, eigenvectors, n_eigenvalues);

  std::cout << "Eigenvalues found:" << std::endl;
  for (unsigned int i = 0; i < eigenvalues.size(); ++i)
    std::cout << "  lambda_" << i << " = " << eigenvalues[i] << std::endl;

  // Output first eigenvector
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  Vector<double> eigenvector_local(n_dofs);
  for (unsigned int i = 0; i < n_dofs; ++i)
    eigenvector_local[i] = eigenvectors[0][i];
  data_out.add_data_vector(eigenvector_local, "eigenmode_0");
  data_out.build_patches();

  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "Eigenvalue problems via SLEPc: -laplacian(u) = lambda*u (step-36 inspired)",
    "tutorial_steps": ["step-36 (SLEPc eigenvalue problem)"],
    "function_space": "FE_Q<dim>(1)",
    "solver": "SLEPc Krylov-Schur (default), Arnoldi, Lanczos, LOBPCG",
    "pitfalls": [
        "Requires deal.II compiled with PETSc + SLEPc support",
        "Generalized eigenvalue: A*x = lambda*M*x where A=stiffness, M=mass",
        "Use AffineConstraints for Dirichlet BCs (distribute to both matrices)",
        "PETSc matrices: PETScWrappers::SparseMatrix, not dealii::SparseMatrix",
        "MPI initialization required: Utilities::MPI::MPI_InitFinalize",
        "For interior eigenvalues: use shift-and-invert spectral transform",
        "Exact eigenvalues on [0,1]^2: lambda_mn = pi^2*(m^2+n^2)",
    ],
}
