"""Hyperelasticity (Neo-Hookean) generator for FEniCSx/dolfinx.

Variants: 3d
"""


KNOWLEDGE = {
    "description": "Nonlinear hyperelasticity (Neo-Hookean) with large deformation",
    "weak_form": "\u03b4\u03a0(u;v) = 0, \u03a0 = \u222b(\u03bc/2)(I_C-3) - \u03bc*ln(J) + (\u03bb/2)(ln(J))\u00b2 dx",
    "function_space": "Vector Lagrange order 1, geometry-nonlinear",
    "solver": "Newton iteration with LU direct solve (SNES newtonls)",
    "pitfalls": [
        "F = I + grad(u), C = F^T F, J = det(F)",
        "Neo-Hookean stored energy must be positive-definite",
        "Large load steps may cause Newton divergence — use load stepping",
        "Locking for nu\u21920.5: use mixed u-p formulations or reduced integration",
        "In dolfinx 0.9+: use NonlinearProblem + NewtonSolver or PETSc SNES",
    ],
}

VARIANTS = ["3d"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate hyperelasticity variant."""
    generators = {
        "3d": _hyperelasticity_3d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(f"Unknown hyperelasticity variant: {variant!r}. Available: {list(generators)}")
    return gen(params)


def _hyperelasticity_3d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    return f'''\
"""Hyperelasticity (Neo-Hookean) — 3D — FEniCSx"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np
from petsc4py import PETSc

domain = mesh.create_box(MPI.COMM_WORLD,
    [[0, 0, 0], [1, 1, 1]], [8, 8, 8], mesh.CellType.tetrahedron)
V = fem.functionspace(domain, ("Lagrange", 1, (3,)))

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

# Fix bottom face
def bottom(x):
    return np.isclose(x[2], 0.0)
bottom_facets = mesh.locate_entities_boundary(domain, fdim, bottom)
bottom_dofs = fem.locate_dofs_topological(V, fdim, bottom_facets)
bc = fem.dirichletbc(np.zeros(3, dtype=default_scalar_type), bottom_dofs, V)

# Prescribed displacement on top
def top(x):
    return np.isclose(x[2], 1.0)
top_facets = mesh.locate_entities_boundary(domain, fdim, top)
top_dofs = fem.locate_dofs_topological(V, fdim, top_facets)
bc_top = fem.dirichletbc(
    np.array([0.0, 0.0, -0.3], dtype=default_scalar_type), top_dofs, V)

# Neo-Hookean material
E_val = {E}
nu_val = {nu}
mu = fem.Constant(domain, default_scalar_type(E_val / (2 * (1 + nu_val))))
lmbda = fem.Constant(domain, default_scalar_type(E_val * nu_val / ((1 + nu_val) * (1 - 2 * nu_val))))

u = fem.Function(V)
v = ufl.TestFunction(V)

d = len(u)
I = ufl.Identity(d)
F = I + ufl.grad(u)
C = F.T * F
J = ufl.det(F)
Ic = ufl.tr(C)

# Stored energy (compressible Neo-Hookean)
psi = (mu / 2) * (Ic - 3) - mu * ufl.ln(J) + (lmbda / 2) * (ufl.ln(J))**2

# First variation (residual)
Pi = psi * ufl.dx
F_form = ufl.derivative(Pi, u, v)

problem = NonlinearProblem(F_form, u, bcs=[bc, bc_top], petsc_options_prefix="hyper",
    petsc_options={{"ksp_type": "preonly", "pc_type": "lu", "pc_factor_mat_solver_type": "mumps",
                   "snes_rtol": 1e-6, "snes_max_it": 25, "snes_monitor": None}})
problem.solve()
n_iters = problem.solver.getIterationNumber()
converged = problem.solver.getConvergedReason() > 0
print(f"Newton: {{n_iters}} iterations, converged={{converged}}")
u.name = "displacement"

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u)

u_arr = u.x.array.reshape(-1, 3)
print(f"max |u| = {{np.linalg.norm(u_arr, axis=1).max():.6e}}")
print(f"DOFs: {{V.dofmap.index_map.size_global * 3}}")
'''
