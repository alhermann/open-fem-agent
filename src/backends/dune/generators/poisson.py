"""DUNE-fem Poisson equation generators and knowledge."""


def _poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Poisson -Δu = f on [0,1]², u=0 on boundary — DUNE-fem with UFL."""
    nx = params.get("nx", 32)
    f_val = params.get("f", 1.0)
    order = params.get("order", 1)
    return f'''\
"""Poisson -Δu = {f_val} on [0,1]², u=0 on boundary — DUNE-fem (UFL)"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import TrialFunction, TestFunction, dot, grad, dx
import numpy as np
import json

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
space = lagrange(gridView, order={order})
u = TrialFunction(space)
v = TestFunction(space)

a = dot(grad(u), grad(v)) * dx
b = {f_val} * v * dx

dbc = DirichletBC(space, 0)
scheme = galerkin([a == b, dbc], solver="cg")
uh = space.interpolate(0, name="solution")
scheme.solve(target=uh)

vals = np.array(uh.as_numpy)
max_val = float(vals.max())
print(f"max(u) = {{max_val:.10f}}")
print(f"DOFs: {{len(vals)}}")

gridView.writeVTK("result", pointdata={{"phi": uh}})

summary = {{"max_value": max_val, "n_dofs": len(vals), "element_type": "Q1 quad"}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("DUNE-fem Poisson solve complete.")
'''


KNOWLEDGE = {
    "poisson": {
        "description": "Poisson with DUNE-fem using UFL (same forms as FEniCS)",
        "solver": "galerkin([a == b, dbc], solver='cg') — Newton-Krylov internally",
        "spaces": "lagrange(gridView, order=k) — Lagrange any order",
        "mesh": "structuredGrid (YaspGrid), ALUGrid (adaptive), Gmsh import",
        "pitfalls": [
            "DUNE uses UFL (same as FEniCS) — weak forms are interchangeable",
            "First run triggers JIT compilation of C++ code — can take 30-60s",
            "Subsequent runs use cached compiled code — much faster",
            "Use dune.ufl.DirichletBC (not dolfinx DirichletBC)",
            "VTK output: gridView.writeVTK('name', pointdata={'field': uh})",
            "structuredGrid creates quad elements, not triangles",
            "Constant(value) needs domain in newer UFL — use scalar directly: 1.0 * v * dx",
        ],
    },
}

GENERATORS = {
    "poisson_2d": _poisson_2d,
}
