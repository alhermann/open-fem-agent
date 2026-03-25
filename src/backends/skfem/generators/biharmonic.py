"""scikit-fem biharmonic / plate bending generators and knowledge."""


def _biharmonic_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Biharmonic equation (Kirchhoff plate bending) with Morley element."""
    refine_level = params.get("refine_level", 4)
    return f'''\
"""Biharmonic equation: Morley element — scikit-fem"""
from skfem import *
import numpy as np
import json

m = MeshTri.init_symmetric().refined({refine_level})
e = ElementTriMorley()
ib = Basis(m, e)

@BilinearForm
def biharmonic(u, v, w):
    # D^2 u : D^2 v (Hessian inner product)
    return (u.hess[0][0]*v.hess[0][0] + u.hess[1][1]*v.hess[1][1]
            + 2*u.hess[0][1]*v.hess[0][1])

@LinearForm
def load(v, w):
    return 1.0 * v

K = asm(biharmonic, ib)
f = asm(load, ib)

# Simply supported: u=0 on boundary
D = ib.get_dofs().flatten()
u = solve(*condense(K, f, D=D))

print(f"Biharmonic: {{K.shape[0]}} DOFs, max(u)={{u.max():.6e}}")

summary = {{"n_dofs": K.shape[0], "max_value": float(u.max())}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Biharmonic solve complete.")
'''


KNOWLEDGE = {
    "biharmonic": {
        "description": "Biharmonic / plate bending (examples 05, 34, 41)",
        "solver": "Direct (4th order system needs fine mesh)",
        "elements": "ElementTriMorley (nonconforming), ElementTriArgyris (C1), ElementQuadBFS (C1 quad)",
        "pitfalls": [
            "Morley: nonconforming plate element, only DOFs at vertices and edge midpoints",
            "Argyris: C1 continuous, 21 DOFs per triangle, 5th degree polynomials",
            "BFS (Bogner-Fox-Schmit): C1 on quads, 16 DOFs per element",
            "Euler-Bernoulli beam (example 34): 1D with ElementLineHermite",
        ],
    },
}

GENERATORS = {
    "biharmonic_2d": _biharmonic_2d,
}
