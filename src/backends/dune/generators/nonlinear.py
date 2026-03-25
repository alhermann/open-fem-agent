"""DUNE-fem nonlinear PDE generators and knowledge."""


def _nonlinear_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Nonlinear PDE via Newton iteration — DUNE-fem."""
    nx = params.get("nx", 32)
    return f'''\
"""Nonlinear PDE: -div((1+u^2)*grad(u)) = 1 — Newton — DUNE-fem"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import TrialFunction, TestFunction, dot, grad, dx
import numpy as np
import json

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
space = lagrange(gridView, order=2)
u = TrialFunction(space)
v = TestFunction(space)

# Nonlinear form: -div((1+u^2)*grad(u)) = f
# DUNE-fem handles Newton automatically when using replace()
from ufl import replace
uh = space.interpolate(0, name="solution")
a = (1 + uh**2) * dot(grad(uh), grad(v)) * dx
b = 1.0 * v * dx

dbc = DirichletBC(space, 0)
scheme = galerkin([a == b, dbc], solver="cg")

# Newton iteration is internal to galerkin scheme
info = scheme.solve(target=uh)
vals = np.array(uh.as_numpy)
print(f"Nonlinear PDE: max={{vals.max():.6f}}")
gridView.writeVTK("result", pointdata={{"solution": uh}})
summary = {{"max_value": float(vals.max()), "n_dofs": len(vals)}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
'''


KNOWLEDGE = {
    "nonlinear": {
        "description": "Nonlinear PDEs solved via built-in Newton iteration",
        "solver": "galerkin scheme handles Newton automatically when form depends on solution",
        "pitfalls": [
            "DUNE-fem linearizes and applies Newton internally — no manual Newton loop needed",
            "For difficult nonlinear problems: use load stepping or continuation",
            "Convergence controlled by scheme parameters (tolerance, max iterations)",
        ],
    },
}

GENERATORS = {
    "nonlinear_2d": _nonlinear_2d,
}
