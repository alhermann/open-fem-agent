"""Biharmonic equation (4th order) generator for FEniCSx/dolfinx.

Variants: 2d
"""


KNOWLEDGE = {
    "description": "Biharmonic equation (4th order PDE) via interior penalty DG",
    "weak_form": "(laplacian(u), laplacian(v))*dx - avg(laplacian(u))*jump(grad(v).n)*dS - jump(grad(u).n)*avg(laplacian(v))*dS + alpha/h*jump(grad(u).n)*jump(grad(v).n)*dS",
    "function_space": "Lagrange order 2+ (C0-IP method), or DG order 2+ (full DG)",
    "solver": {"ksp_type": "preonly", "pc_type": "lu"},
    "pitfalls": [
        "4th order PDE: standard C0 Lagrange cannot represent it directly",
        "C0-interior penalty (C0-IP): uses P2 continuous elements with jump penalties",
        "Full DG: uses DG elements with all interface terms (more DOFs)",
        "Penalty parameter alpha must be large enough for coercivity (typ. 4-16)",
        "Clamped BC: u=0 AND grad(u).n=0; simply supported: u=0 AND laplacian(u)=0",
        "For Kirchhoff plates: biharmonic in displacement w, load q",
    ],
}

VARIANTS = ["2d"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate biharmonic variant."""
    generators = {
        "2d": _biharmonic_2d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(f"Unknown biharmonic variant: {variant!r}. Available: {list(generators)}")
    return gen(params)


def _biharmonic_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable script. All parameter defaults are placeholders. The user/agent must set values appropriate to the specific problem being solved."""
    nx = params.get("nx", 32)
    ny = params.get("ny", 32)
    penalty = params.get("penalty", 8.0)
    return f'''\
"""Biharmonic equation on unit square — interior penalty DG — FEniCSx/dolfinx
laplacian(laplacian(u)) = f on [0,1]^2
u = 0, grad(u).n = 0 on boundary
Symmetric interior penalty (C0-IP or full DG) method.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
import ufl
import numpy as np
from petsc4py import PETSc

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
V = fem.functionspace(domain, ("Lagrange", 2))

# Boundary conditions: u = 0 and du/dn = 0 on all boundaries
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)
boundary_facets = mesh.exterior_facet_indices(domain.topology)
dofs = fem.locate_dofs_topological(V, fdim, boundary_facets)
bc = fem.dirichletbc(default_scalar_type(0.0), dofs, V)

# Source term
x = ufl.SpatialCoordinate(domain)
f_expr = fem.Constant(domain, default_scalar_type(1.0))

# Interior penalty DG bilinear form for biharmonic
u = ufl.TrialFunction(V)
v = ufl.TestFunction(V)
n = ufl.FacetNormal(domain)
h = ufl.CellDiameter(domain)
h_avg = (h("+") + h("-")) / 2.0
alpha = fem.Constant(domain, default_scalar_type({penalty}))

# Biharmonic via C0-interior penalty:
# Volume: (laplacian(u), laplacian(v))
# Interior facets: penalty jumps in normal derivatives
a = ufl.inner(ufl.div(ufl.grad(u)), ufl.div(ufl.grad(v))) * ufl.dx \\
    - ufl.inner(ufl.avg(ufl.div(ufl.grad(u))), ufl.jump(ufl.grad(v), n)) * ufl.dS \\
    - ufl.inner(ufl.jump(ufl.grad(u), n), ufl.avg(ufl.div(ufl.grad(v)))) * ufl.dS \\
    + alpha / h_avg * ufl.inner(ufl.jump(ufl.grad(u), n), ufl.jump(ufl.grad(v), n)) * ufl.dS

L = f_expr * v * ufl.dx

# Assemble and solve
a_form = fem.form(a)
L_form = fem.form(L)
from dolfinx.fem.petsc import assemble_matrix, assemble_vector, apply_lifting, set_bc

A = assemble_matrix(a_form, bcs=[bc])
A.assemble()
b = assemble_vector(L_form)
apply_lifting(b, [a_form], bcs=[[bc]])
b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
set_bc(b, [bc])

# Solve with direct solver
solver = PETSc.KSP().create(domain.comm)
solver.setOperators(A)
solver.setType(PETSc.KSP.Type.PREONLY)
solver.getPC().setType(PETSc.PC.Type.LU)
uh = fem.Function(V)
uh.name = "u"
solver.solve(b, uh.x.petsc_vec)

# Output
from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "result.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(uh)

u_array = uh.x.array
print(f"Biharmonic solved: min(u)={{u_array.min():.6e}}, max(u)={{u_array.max():.6e}}")
print(f"DOFs: {{V.dofmap.index_map.size_global}}")
'''
