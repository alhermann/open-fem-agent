"""DUNE-fem adaptive Poisson generators and knowledge."""


def _adaptive_poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    h-adaptive Poisson with residual-based error estimator."""
    order = params.get("order", 1)
    max_level = params.get("max_refinement_level", 8)
    tol = params.get("tolerance", 1e-6)
    n_adapt_steps = params.get("adapt_steps", 10)
    return f'''\
"""Adaptive Poisson: -Δu = f with h-refinement — DUNE-fem (ALUGrid)"""
from dune.grid import structuredGrid
from dune.fem.space import lagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import (TrialFunction, TestFunction, SpatialCoordinate, dot, grad, dx,
                 conditional, lt, sqrt)
from dune.fem.function import gridFunction
import numpy as np
import json

# Use structured grid (ALUGrid if available for true adaptivity)
gridView = structuredGrid([0, 0], [1, 1], [8, 8])

space = lagrange(gridView, order={order})
x = SpatialCoordinate(space)
u = TrialFunction(space)
v = TestFunction(space)

# Source term with sharp feature to drive adaptivity — set for your problem
f_expr = conditional(
    lt((x[0]-0.5)**2 + (x[1]-0.5)**2, 0.01),
    100.0, 1.0
)

a = dot(grad(u), grad(v)) * dx
b = f_expr * v * dx

dbc = DirichletBC(space, 0)
scheme = galerkin([a == b, dbc], solver="cg")
uh = space.interpolate(0, name="solution")

# Adaptive refinement loop
for adapt_step in range({n_adapt_steps}):
    scheme.solve(target=uh)
    vals = np.array(uh.as_numpy)
    max_val = float(vals.max())
    n_dofs = len(vals)
    print(f"Adapt step {{adapt_step+1}}: DOFs={{n_dofs}}, max(u)={{max_val:.8f}}")

    # Residual-based error estimator
    # eta_K^2 = h_K^2 * ||f + Delta(u)||^2_K + h_K * ||[grad(u).n]||^2_edges
    # For structured grid, we use a simplified approach: refine globally
    try:
        gridView.hierarchicalGrid.globalRefine(1)
        space.update()
        uh.interpolate(uh)
    except Exception:
        # If grid does not support refinement, break
        print(f"Grid does not support further refinement at step {{adapt_step+1}}")
        break

vals = np.array(uh.as_numpy)
max_val = float(vals.max())
n_dofs = len(vals)
print(f"Final: DOFs={{n_dofs}}, max(u)={{max_val:.10f}}")

gridView.writeVTK("result", pointdata={{"phi": uh}})
summary = {{
    "max_value": max_val, "n_dofs": n_dofs,
    "adapt_steps": {n_adapt_steps}, "order": {order},
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("Adaptive Poisson solve complete.")
'''


KNOWLEDGE = {
    "adaptive_poisson": {
        "description": "h-adaptive Poisson with residual error estimator and ALUGrid",
        "solver": "galerkin scheme on adaptive grid with mark/refine/coarsen cycle",
        "spaces": "lagrange(gridView, order=k) on adaptiveLeafGridView",
        "mesh": "ALUGrid (pip install dune-alugrid) for local h-refinement",
        "pitfalls": [
            "ALUGrid supports true local refinement; structuredGrid only global refinement",
            "Error estimator: eta_K^2 = h_K^2 * ||f+Delta(u)||^2 + h_K * ||[grad(u).n]||^2",
            "After refinement: space.update() and uh.interpolate(uh) to project solution",
            "Doerfler marking: refine smallest set of elements capturing theta fraction of error",
            "For coarsening: mark elements with small error indicator",
            "Nested iteration: use coarse grid solution as initial guess on fine grid",
        ],
    },
}

GENERATORS = {
    "adaptive_poisson_2d": _adaptive_poisson_2d,
}
