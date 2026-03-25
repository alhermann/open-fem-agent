"""NGSolve Poisson equation generators and knowledge."""


def _poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Poisson -Δu = f on [0,1]², u=0 on ∂Ω."""
    nx = params.get("nx", 32)
    f_val = params.get("f", 1.0)
    order = params.get("order", 1)
    maxh = 1.0 / nx
    return f'''\
"""Poisson -Δu = {f_val} on [0,1]², u=0 on boundary — NGSolve"""
from ngsolve import *
from ngsolve.webgui import Draw  # type: ignore
import json

# Mesh
mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))
print(f"Mesh: {{mesh.ne}} elements, {{mesh.nv}} vertices")

# FE space
fes = H1(mesh, order={order}, dirichlet="bottom|right|top|left")
u, v = fes.TnT()

# Bilinear form and linear form
a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
f = LinearForm({f_val}*v*dx).Assemble()

# Solve
gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

# Output
max_val = max(gfu.vec)
print(f"max(u) = {{max_val:.10f}}")
print(f"DOFs: {{fes.ndof}}")

# VTK output
vtk = VTKOutput(mesh, coefs=[gfu], names=["phi"],
                filename="result", subdivision=0)
vtk.Do()

# Summary
summary = {{
    "max_value": float(max_val),
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
    "h": {maxh},
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Poisson solve complete.")
'''


def _poisson_3d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Poisson -Δu = f on [0,1]³, u=0 on ∂Ω."""
    nx = params.get("nx", 8)
    f_val = params.get("f", 1.0)
    order = params.get("order", 1)
    maxh = 1.0 / nx
    return f'''\
"""Poisson -Δu = {f_val} on [0,1]³, u=0 on boundary — NGSolve"""
from ngsolve import *
import json
from netgen.csg import unit_cube

# Mesh
geo = unit_cube
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))
print(f"Mesh: {{mesh.ne}} elements, {{mesh.nv}} vertices")

# FE space
fes = H1(mesh, order={order}, dirichlet=".*")
u, v = fes.TnT()

a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
f = LinearForm({f_val}*v*dx).Assemble()

gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

max_val = max(gfu.vec)
print(f"max(u) = {{max_val:.10f}}")

vtk = VTKOutput(mesh, coefs=[gfu], names=["phi"],
                filename="result", subdivision=0)
vtk.Do()

summary = {{
    "max_value": float(max_val),
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("3D Poisson solve complete.")
'''


KNOWLEDGE = {
    "poisson": {
        "description": "Poisson equation -Δu = f with NGSolve (arbitrary-order H1)",
        "spaces": "H1 (Lagrange, order 1-10+)",
        "solver": "Direct: sparsecholesky, umfpack, pardiso. Iterative: CG + h1amg/multigrid/bddc",
        "mesh": "unit_square, unit_cube, SplineGeometry (2D), CSG (3D), OCC (CAD import)",
        "pitfalls": [
            "Boundary names must exactly match mesh labels (case-sensitive): 'left', 'right', 'top', 'bottom'",
            "max(gfu.vec) gives max over DOF values, not pointwise max",
            "Dirichlet BCs: set via gfu.Set() then modify RHS: f.vec -= a.mat * gfu.vec",
            "VTKOutput writes .vtu natively — no XDMF conversion needed",
            "Use subdivision=2 for higher-order visualization in ParaView",
        ],
    },
}

GENERATORS = {
    "poisson_2d": _poisson_2d,
    "poisson_3d": _poisson_3d,
}
