"""scikit-fem linear elasticity generators and knowledge."""


def _elasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Linear elasticity on rectangular domain, fixed left edge, body force."""
    nx = params.get("nx", 40)
    ny = params.get("ny", 4)
    lx = params.get("lx", 10.0)
    ly = params.get("ly", 1.0)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    return f'''\
"""Linear elasticity: rectangular domain, fixed left — scikit-fem"""
from skfem import *
from skfem.models.elasticity import linear_elasticity, lame_parameters
import numpy as np
import json

m = MeshQuad.init_tensor(np.linspace(0, {lx}, {nx+1}), np.linspace(0, {ly}, {ny+1}))
e = ElementVector(ElementQuad1())
ib = Basis(m, e)

lam, mu = lame_parameters({E}, {nu})
K = linear_elasticity(lam, mu).assemble(ib)

# Body force — set for your problem
@LinearForm
def body_force(v, w):
    return -1.0 * v[1]

f = body_force.assemble(ib)

# Fix left edge
D = ib.get_dofs("left").flatten()
u = solve(*condense(K, f, D=D))

# Tip displacement
u_reshaped = u.reshape(2, -1)
max_uy = u_reshaped[1].min()
print(f"Max tip displacement: {{max_uy:.6f}}")

summary = {{"max_displacement_y": float(max_uy), "n_dofs": int(K.shape[0])}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "linear_elasticity": {
        "description": "Linear elasticity: plane strain/stress, 3D (examples 02, 03, 11, 21)",
        "solver": "Direct sparse (scipy.sparse.linalg.spsolve)",
        "elements": "ElementVector(ElementQuad1()) or ElementVector(ElementTriP1())",
        "built_in_forms": "linear_elasticity, linear_stress (from skfem.models.elasticity)",
        "pitfalls": [
            "Use ElementVector(ElementQuad1()) for vector problems, NOT ElementQuad1() alone",
            "lame_parameters(E, nu) computes lambda and mu from engineering constants",
            "linear_elasticity(lam, mu) returns a BilinearForm directly",
            "For eigenvalue problems (vibration): use eigsh(K, M=M, k=n, sigma=0)",
        ],
    },
}

GENERATORS = {
    "linear_elasticity_2d": _elasticity_2d,
}
