"""DUNE-fem DG advection generators and knowledge."""


def _dg_advection_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    DG method for pure advection equation with upwind flux."""
    nx = params.get("nx", 32)
    order = params.get("order", 1)
    dt = params.get("dt", 0.001)
    T_end = params.get("T_end", 0.5)
    return f'''\
"""DG advection: du/dt + b.grad(u) = 0 — upwind flux — DUNE-fem"""
from dune.grid import structuredGrid
from dune.fem.space import dglagrange
from dune.fem.scheme import galerkin
from dune.ufl import DirichletBC
from ufl import (TrialFunction, TestFunction, SpatialCoordinate, dot, grad, dx,
                 conditional, lt, FacetNormal, avg, jump, dS, ds,
                 CellDiameter)
import numpy as np
import json

gridView = structuredGrid([0, 0], [1, 1], [{nx}, {nx}])
space = dglagrange(gridView, order={order})
x = SpatialCoordinate(space)

# Initial condition: smooth bump — set for your problem
u_n = space.interpolate(
    conditional(lt((x[0]-0.25)**2 + (x[1]-0.5)**2, 0.04), 1.0, 0.0),
    name="u"
)

u = TrialFunction(space)
v = TestFunction(space)
n = FacetNormal(space)
h = CellDiameter(space)

# Advection velocity — set for your problem
b0, b1 = 1.0, 0.5

# Bilinear form: mass + dt * advection with upwind flux
dt = {dt}

# Mass term
a_mass = u * v * dx

# Advection: volume term + upwind flux on interior facets + inflow boundary
bn = b0 * n[0] + b1 * n[1]

# For DG advection we use an explicit Euler approach with the galerkin solver
# Volume advection
a_adv = -(b0 * u * v.dx(0) + b1 * u * v.dx(1)) * dx

# Interior facet upwind flux
from ufl import gt
u_up = conditional(gt(bn("+"), 0), u("+"), u("-"))
a_adv += (bn("+") * u_up * (v("+") - v("-"))) * dS

# Boundary outflow
a_adv += conditional(gt(bn, 0), bn * u * v, 0) * ds

# Combined form: u_new = u_old - dt * advection(u_old)
a = a_mass - dt * a_adv
b_form = u_n * v * dx

scheme = galerkin([a == b_form], solver="cg")

# Time stepping
t = 0.0
n_steps = int({T_end} / dt)
for step in range(n_steps):
    scheme.solve(target=u_n)
    t += dt

vals = np.array(u_n.as_numpy)
print(f"DG advection: t={{t:.4f}}, max={{vals.max():.6f}}, min={{vals.min():.6f}}")
gridView.writeVTK("result", pointdata={{"concentration": u_n}})
summary = {{
    "time": t, "max_value": float(vals.max()), "min_value": float(vals.min()),
    "n_dofs": len(vals), "order": {order}, "dt": dt,
}}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("DG advection solve complete.")
'''


KNOWLEDGE = {
    "dg_advection": {
        "description": "DG method for pure advection equations (upwind flux, explicit time stepping)",
        "solver": "Explicit Euler with DG spatial discretization, upwind numerical flux",
        "spaces": "dglagrange(gridView, order=k) — discontinuous Lagrange of any order",
        "time_stepping": "Explicit Euler, SSP-RK2, SSP-RK3 for stability",
        "pitfalls": [
            "CFL condition: dt < h / (2*order+1) / max(|b|) for stability",
            "Upwind flux: use conditional(gt(b.n, 0), u('+'), u('-')) for upwind selection",
            "Interior facet integrals: dS in UFL (capital S), boundary: ds (lowercase)",
            "DG with dune-fem-dg module: optimized SSP-RK time steppers available",
            "For high-order DG: use modal basis (dgonb, dglegendre) for better conditioning",
            "Limiters may be needed for discontinuous solutions (TVD/TVB/WENO)",
        ],
    },
}

GENERATORS = {
    "dg_advection_2d": _dg_advection_2d,
}
