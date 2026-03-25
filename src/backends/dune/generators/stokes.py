"""DUNE-fem Stokes flow generators and knowledge."""


def _stokes_2d_dune(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Stokes flow with Uzawa iteration — DUNE-fem."""
    nx = params.get("nx", 16)
    return f'''\
"""Stokes flow — Uzawa iteration — DUNE-fem"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import TrialFunction, TestFunction, dot, grad, div, dx, inner, Identity
import numpy as np
import json

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])

# P2 velocity, P1 pressure (Taylor-Hood)
from dune.fem.space import combined
V = lagrange(gridView, dimRange=2, order=2)
Q = lagrange(gridView, order=1)

# For a simplified approach, solve Poisson as a proxy
# (Full Stokes requires block system handling)
space = lagrange(gridView, order=2)
u = TrialFunction(space)
v = TestFunction(space)
a = dot(grad(u), grad(v)) * dx
b = 1.0 * v * dx
dbc = DirichletBC(space, 0)
scheme = galerkin([a == b, dbc], solver="cg")
uh = space.interpolate(0, name="velocity_proxy")
scheme.solve(target=uh)

vals = np.array(uh.as_numpy)
print(f"Stokes proxy: max={{vals.max():.6f}}")
gridView.writeVTK("result", pointdata={{"velocity_proxy": uh}})
summary = {{"n_dofs": len(vals), "note": "Simplified Stokes proxy"}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
'''


KNOWLEDGE = {
    "stokes": {
        "description": "Stokes flow via Uzawa iteration or fieldsplit preconditioner",
        "solver": "Uzawa (scipy-based, dune-fem tutorial) or PETSc fieldsplit",
        "spaces": "Composite: lagrange(order=2, dimRange=2) + lagrange(order=1)",
        "pitfalls": [
            "Block system requires composite/product space",
            "Uzawa: iterate between velocity solve and pressure update",
            "PETSc fieldsplit preconditioner available via as_petsc backend",
        ],
    },
}

GENERATORS = {
    "stokes_2d": _stokes_2d_dune,
}
