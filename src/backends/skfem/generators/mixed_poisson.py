"""scikit-fem mixed Poisson generators and knowledge."""


def _mixed_poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Mixed Poisson with Raviart-Thomas + piecewise constant."""
    nx = params.get("nx", 16)
    refine_level = params.get("refine_level", 4)
    return f'''\
"""Mixed Poisson: RT1 + P0 — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace
import numpy as np
import json

m = MeshTri.init_symmetric().refined({refine_level})
e_rt = ElementTriRT0()
e_dg = ElementTriP0()

ib_rt = Basis(m, e_rt)
ib_dg = Basis(m, e_dg)

@BilinearForm
def mass_rt(sigma, tau, w):
    return sigma[0]*tau[0] + sigma[1]*tau[1]

@BilinearForm
def div_form(sigma, v, w):
    return (sigma[0].grad[0] + sigma[1].grad[1]) * v

A = asm(mass_rt, ib_rt)
B = asm(div_form, ib_rt, ib_dg)

from scipy.sparse import bmat
K = bmat([[A, B.T], [B, None]], format='csr')
f = np.zeros(K.shape[0])
# Source in scalar part
f[A.shape[0]:] = -1.0 * asm(LinearForm(lambda v, w: 1.0 * v), ib_dg)

u = np.linalg.lstsq(K.toarray(), f, rcond=None)[0]
print(f"Mixed Poisson: {{K.shape[0]}} DOFs")

summary = {{"n_dofs": K.shape[0], "n_elements": m.nelements}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Mixed Poisson solve complete.")
'''


KNOWLEDGE = {
    "mixed_poisson": {
        "description": "Mixed Poisson with Raviart-Thomas (example 37)",
        "solver": "Direct (saddle-point system) or iterative with Schur complement",
        "elements": "ElementTriRT0 (flux) + ElementTriP0 (scalar)",
        "pitfalls": [
            "Block system: [[A, B^T], [B, 0]] where A = mass(sigma,tau), B = div(sigma)*v",
            "RT elements: normal component continuous, divergence well-defined",
            "For Neumann BC: add boundary integral to RHS",
        ],
    },
}

GENERATORS = {
    "mixed_poisson_2d": _mixed_poisson_2d,
}
