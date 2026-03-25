"""Linear elasticity templates for deal.II.

Based on deal.II tutorial step-8.
"""


def _elasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    Based on deal.II step-8.
    """
    refinements = params.get("refinements", 4)
    E_val = params.get("E", 1000.0)
    nu_val = params.get("nu", 0.3)
    lx = params.get("lx", 10.0)
    ly = params.get("ly", 1.0)
    nx_cells = int(lx * 4)
    ny_cells = max(int(ly * 4), 1)
    return f'''\
/* Linear elasticity — based on deal.II step-8
 * 2D plane strain, fixed left edge, body force pointing down
 */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/component_mask.h>
#include <fstream>
#include <iostream>

using namespace dealii;

// Body force — set direction and magnitude for your problem
template <int dim>
class BodyForce : public Function<dim>
{{
public:
  BodyForce() : Function<dim>(dim) {{}}
  virtual void vector_value(const Point<dim> &, Vector<double> &values) const override
  {{
    values    = 0;
    values[1] = -1.0; // downward
  }}
}};

int main()
{{
  const int dim = 2;

  // Domain
  Triangulation<dim> triangulation;
  GridGenerator::subdivided_hyper_rectangle(triangulation,
    {{{nx_cells}u, {ny_cells}u}}, Point<dim>(0, 0), Point<dim>({lx}, {ly}), true /*colorize*/);

  FESystem<dim> fe(FE_Q<dim>(1), dim);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  std::cout << "Number of DOFs: " << dof_handler.n_dofs() << std::endl;

  // Sparsity
  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);
  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> system_rhs(dof_handler.n_dofs());

  // Material
  const double E  = {E_val};
  const double nu = {nu_val};
  const double mu     = E / (2.0 * (1.0 + nu));
  const double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

  // Assemble
  QGauss<dim> quadrature(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature,
    update_values | update_gradients | update_quadrature_points | update_JxW_values);

  const unsigned int dpc = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dpc, dpc);
  Vector<double>     cell_rhs(dpc);
  std::vector<types::global_dof_index> local_dof_indices(dpc);

  BodyForce<dim> body_force;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {{
      fe_values.reinit(cell);
      cell_matrix = 0;
      cell_rhs    = 0;

      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {{
          // Body force at quadrature point
          Vector<double> f_val(dim);
          body_force.vector_value(fe_values.quadrature_point(q), f_val);

          for (unsigned int i = 0; i < dpc; ++i)
            {{
              const unsigned int ci = fe.system_to_component_index(i).first;

              for (unsigned int j = 0; j < dpc; ++j)
                {{
                  const unsigned int cj = fe.system_to_component_index(j).first;

                  cell_matrix(i, j) +=
                    (fe_values.shape_grad(i, q)[ci] *
                     fe_values.shape_grad(j, q)[cj] * lambda
                     +
                     fe_values.shape_grad(i, q) *
                     fe_values.shape_grad(j, q) *
                     (ci == cj ? mu : 0.0)
                     +
                     fe_values.shape_grad(i, q)[cj] *
                     fe_values.shape_grad(j, q)[ci] * mu
                    ) * fe_values.JxW(q);
                }}

              cell_rhs(i) += fe_values.shape_value(i, q) * f_val[ci] *
                             fe_values.JxW(q);
            }}
        }}

      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dpc; ++i)
        {{
          for (unsigned int j = 0; j < dpc; ++j)
            system_matrix.add(local_dof_indices[i],
                              local_dof_indices[j],
                              cell_matrix(i, j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }}
    }}

  // BC: fix left edge (x=0), all components
  std::map<types::global_dof_index, double> boundary_values;
  // Left boundary = id 0 for hyper_rectangle
  VectorTools::interpolate_boundary_values(dof_handler,
    0, Functions::ZeroFunction<dim>(dim), boundary_values);
  MatrixTools::apply_boundary_values(boundary_values,
    system_matrix, solution, system_rhs);

  // Solve
  SolverControl solver_control(5000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "Solver converged in " << solver_control.last_step()
            << " iterations." << std::endl;

  // Output
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  std::vector<std::string> names = {{"ux", "uy"}};
  data_out.add_data_vector(solution, names);
  data_out.build_patches();

  std::ofstream output("result.vtu");
  data_out.write_vtu(output);

  std::cout << "Output written to result.vtu" << std::endl;
  return 0;
}}
'''


def _elasticity_thick_beam(params: dict) -> str:
    """FORMAT TEMPLATE: generates a compilable deal.II C++ program.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    lx = params.get("lx", 5.0)
    ly = params.get("ly", 2.0)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    nx = int(lx * 8)
    ny = int(ly * 8)
    return f'''\
/* Linear elasticity on {lx}x{ly} domain — deal.II
 * Fixed left edge, body force.
 */
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/fe/fe_values.h>
#include <fstream>
#include <iostream>

using namespace dealii;

template <int dim>
class BodyForce : public Function<dim>
{{
public:
  BodyForce() : Function<dim>(dim) {{}}
  virtual void vector_value(const Point<dim> &, Vector<double> &values) const override
  {{
    values = 0;
    values[1] = -1.0;
  }}
}};

int main()
{{
  const int dim = 2;
  Triangulation<dim> triangulation;
  GridGenerator::subdivided_hyper_rectangle(triangulation,
    {{{nx}u, {ny}u}}, Point<dim>(0, 0), Point<dim>({lx}, {ly}), true /*colorize*/);

  FESystem<dim> fe(FE_Q<dim>(1), dim);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  std::cout << "DOFs: " << dof_handler.n_dofs() << std::endl;

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler, dsp);
  SparsityPattern sparsity_pattern;
  sparsity_pattern.copy_from(dsp);

  SparseMatrix<double> system_matrix;
  system_matrix.reinit(sparsity_pattern);
  Vector<double> solution(dof_handler.n_dofs());
  Vector<double> system_rhs(dof_handler.n_dofs());

  const double E  = {E};
  const double nu = {nu};
  const double mu     = E / (2.0 * (1.0 + nu));
  const double lambda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));

  QGauss<dim> quadrature(fe.degree + 1);
  FEValues<dim> fe_values(fe, quadrature,
    update_values | update_gradients | update_quadrature_points | update_JxW_values);

  const unsigned int dpc = fe.n_dofs_per_cell();
  FullMatrix<double> cell_matrix(dpc, dpc);
  Vector<double> cell_rhs(dpc);
  std::vector<types::global_dof_index> local_dof_indices(dpc);
  BodyForce<dim> body_force;

  for (const auto &cell : dof_handler.active_cell_iterators())
    {{
      fe_values.reinit(cell);
      cell_matrix = 0; cell_rhs = 0;
      for (unsigned int q = 0; q < quadrature.size(); ++q)
        {{
          Vector<double> f_val(dim);
          body_force.vector_value(fe_values.quadrature_point(q), f_val);
          for (unsigned int i = 0; i < dpc; ++i)
            {{
              const unsigned int ci = fe.system_to_component_index(i).first;
              for (unsigned int j = 0; j < dpc; ++j)
                {{
                  const unsigned int cj = fe.system_to_component_index(j).first;
                  cell_matrix(i, j) +=
                    (fe_values.shape_grad(i, q)[ci] *
                     fe_values.shape_grad(j, q)[cj] * lambda
                     + fe_values.shape_grad(i, q) *
                       fe_values.shape_grad(j, q) * (ci == cj ? mu : 0.0)
                     + fe_values.shape_grad(i, q)[cj] *
                       fe_values.shape_grad(j, q)[ci] * mu
                    ) * fe_values.JxW(q);
                }}
              cell_rhs(i) += fe_values.shape_value(i, q) * f_val[ci] * fe_values.JxW(q);
            }}
        }}
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int i = 0; i < dpc; ++i)
        {{
          for (unsigned int j = 0; j < dpc; ++j)
            system_matrix.add(local_dof_indices[i], local_dof_indices[j], cell_matrix(i, j));
          system_rhs(local_dof_indices[i]) += cell_rhs(i);
        }}
    }}

  std::map<types::global_dof_index, double> boundary_values;
  VectorTools::interpolate_boundary_values(dof_handler, 0,
    Functions::ZeroFunction<dim>(dim), boundary_values);
  MatrixTools::apply_boundary_values(boundary_values, system_matrix, solution, system_rhs);

  SolverControl solver_control(5000, 1e-12);
  SolverCG<Vector<double>> solver(solver_control);
  solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity());

  std::cout << "Solver: " << solver_control.last_step() << " iterations" << std::endl;

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  std::vector<std::string> names = {{"ux", "uy"}};
  data_out.add_data_vector(solution, names);
  data_out.build_patches();
  std::ofstream output("result.vtu");
  data_out.write_vtu(output);
  return 0;
}}
'''


# ── Knowledge ────────────────────────────────────────────────────────────

KNOWLEDGE = {
    "description": "Linear elasticity (step-8, step-17 parallel, step-18 quasi-static)",
    "tutorial_steps": ["step-8 (basic)", "step-17 (MPI parallel)", "step-18 (incremental)"],
    "function_space": "FESystem<dim>(FE_Q<dim>(1), dim) — vector Lagrange",
    "solver": "CG + PreconditionSSOR (serial), SolverCG + BoomerAMG (parallel)",
    "pitfalls": [
        "Use FEValuesExtractors::Vector(0) for velocity-like access in assembly",
        "Lame parameters: mu = E/(2(1+nu)), lambda = E*nu/((1+nu)(1-2nu))",
        "For plane stress: modify lambda to lambda_star = 2*mu*lambda/(2*mu+lambda). "
        "Code: double lam_star = 2*mu*lam / (2*mu + lam);",
        "Body force: add to cell_rhs via fe_values[velocities].value(i,q)",
        "deal.II 2D ONLY reads QUADS from Gmsh — no triangles. Always use "
        "gmsh.option.setNumber('Mesh.RecombineAll', 1) to produce quads.",
        "Gmsh element order != FE polynomial degree. ALWAYS use first-order "
        "geometry elements in Gmsh (default). The FE degree (Q1, Q2) "
        "is set in the C++ code via FE_Q<dim>(degree). Do NOT set "
        "Mesh.ElementOrder=2 in Gmsh — deal.II cannot read second-order "
        "geometry elements (Tri6, Quad9).",
    ],
}
