"""DUNE-fem heat conduction generators and knowledge."""


def _heat_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Heat conduction on [0,1]² — DUNE-fem."""
    nx = params.get("nx", 32)
    T_left = params.get("T_left", 100.0)
    T_right = params.get("T_right", 0.0)
    return f'''\
"""Heat conduction on [0,1]² — DUNE-fem"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import TrialFunction, TestFunction, SpatialCoordinate, dot, grad, dx, conditional, lt
import numpy as np
import json

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
space = lagrange(gridView, order=1)
x = SpatialCoordinate(space)
u = TrialFunction(space)
v = TestFunction(space)

a = dot(grad(u), grad(v)) * dx
b = 0 * v * dx  # No source

# Dirichlet BCs — set for your problem
bc_expr = conditional(lt(x[0], 0.01), {T_left}, conditional(lt(1.0 - x[0], 0.01), {T_right}, 0.0))
dbc = DirichletBC(space, bc_expr)

scheme = galerkin([a == b, dbc], solver="cg")
uh = space.interpolate(0, name="temperature")
scheme.solve(target=uh)

vals = np.array(uh.as_numpy)
print(f"Temperature: max={{vals.max():.6f}}")

gridView.writeVTK("result", pointdata={{"temperature": uh}})

summary = {{"max_value": float(vals.max()), "n_dofs": len(vals)}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
'''


KNOWLEDGE = {
    "heat": {
        "description": "Heat conduction: steady and transient (backward Euler, Crank-Nicolson)",
        "solver": "Same galerkin scheme; for transient, use time-stepping loop",
        "time_stepping": "Backward Euler, Crank-Nicolson, DIRK23, DIRK34, SDIRK22, Heun",
        "pitfalls": [
            "Non-homogeneous Dirichlet via conditional() in UFL expression",
            "Transient: mass matrix + dt*stiffness matrix per step",
            "DUNE caches compiled code — first step slow, rest fast",
        ],
    },
}

GENERATORS = {
    "heat_2d": _heat_2d,
}
