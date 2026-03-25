"""Convection-diffusion (SUPG) generator for FEniCSx/dolfinx.

Variants: 2d
"""


KNOWLEDGE = {
    "description": "Convection-diffusion with SUPG stabilization",
    "weak_form": "(eps*grad(u) + b*u, grad(v))*dx + tau*(b\u00b7grad(u), b\u00b7grad(v))*dx = (f, v+tau*b\u00b7grad(v))*dx",
    "function_space": "Lagrange order 1",
    "solver": {"ksp_type": "preonly", "pc_type": "lu"},
    "pitfalls": [
        "SUPG: tau = h/(2|b|) * (coth(Pe) - 1/Pe), Pe = |b|*h/(2*eps)",
        "Without stabilization: oscillations for Peclet > 1",
        "CellDiameter(domain) gives the element size h for tau computation",
        "For pure advection (eps=0): use DG methods instead",
    ],
}

VARIANTS = ["2d"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate convection-diffusion variant."""
    generators = {
        "2d": _convection_diffusion_2d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(f"Unknown convection_diffusion variant: {variant!r}. Available: {list(generators)}")
    return gen(params)


def _convection_diffusion_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable FEniCSx script.

    All parameter defaults are placeholders. The user/agent must set values
    appropriate to the specific problem being solved.
    """
    nx = params.get("nx", 32)
    eps = params.get("diffusion", 0.01)
    return f'''\
"""Convection-diffusion: SUPG stabilized — FEniCSx"""
from mpi4py import MPI
from dolfinx import mesh, fem, default_scalar_type
from dolfinx.fem.petsc import LinearProblem
import ufl
import numpy as np

domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {nx}, mesh.CellType.triangle)
gdim = domain.geometry.dim
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

V = fem.functionspace(domain, ("Lagrange", 1))

def boundary(x):
    return np.isclose(x[0], 0) | np.isclose(x[0], 1) | np.isclose(x[1], 0) | np.isclose(x[1], 1)
bc_facets = mesh.locate_entities_boundary(domain, fdim, boundary)
bc = fem.dirichletbc(default_scalar_type(0),
    fem.locate_dofs_topological(V, fdim, bc_facets), V)

u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
b = ufl.as_vector([1.0, 0.5])
eps = {eps}
f_src = fem.Constant(domain, default_scalar_type(1.0))

# Standard Galerkin + SUPG stabilization
h = ufl.CellDiameter(domain)
Pe = ufl.sqrt(ufl.dot(b, b)) * h / (2.0 * eps)
tau = h / (2.0 * ufl.sqrt(ufl.dot(b, b))) * (1.0 / ufl.tanh(Pe) - 1.0 / Pe)

a = (eps * ufl.dot(ufl.grad(u), ufl.grad(v)) + ufl.dot(b, ufl.grad(u)) * v) * ufl.dx
a += tau * ufl.dot(b, ufl.grad(u)) * ufl.dot(b, ufl.grad(v)) * ufl.dx
L = f_src * v * ufl.dx + tau * f_src * ufl.dot(b, ufl.grad(v)) * ufl.dx

problem = LinearProblem(a, L, bcs=[bc],
    petsc_options_prefix="cd",
    petsc_options={{"ksp_type": "preonly", "pc_type": "lu"}})
uh = problem.solve()
uh.name = "concentration"

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

print(f"max(u) = {{uh.x.array.max():.6f}}")
print("Convection-diffusion solve complete.")
'''
