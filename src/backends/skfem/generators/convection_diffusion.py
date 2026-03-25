"""scikit-fem convection-diffusion generators and knowledge."""


def _convdiff_2d_skfem(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Convection-diffusion with SUPG-like stabilization."""
    nx = params.get("nx", 32)
    eps = params.get("diffusion", 0.01)
    return f'''\
"""Convection-diffusion: stabilized — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace
import numpy as np
import json

m = MeshQuad.init_tensor(np.linspace(0, 1, {nx+1}), np.linspace(0, 1, {nx+1}))
e = ElementQuad1()
ib = Basis(m, e)

eps = {eps}
b = np.array([1.0, 0.5])

@BilinearForm
def advdiff(u, v, w):
    return eps * (u.grad[0]*v.grad[0] + u.grad[1]*v.grad[1]) + (b[0]*u.grad[0] + b[1]*u.grad[1]) * v

K = asm(advdiff, ib)
f = np.ones(K.shape[0])  # unit source

D = ib.get_dofs().flatten()
u = solve(*condense(K, f, D=D))

print(f"ConvDiff: max(u) = {{u.max():.6f}}")
summary = {{"max_value": float(u.max()), "n_dofs": K.shape[0], "diffusion": eps}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Convection-diffusion solve complete.")
'''


KNOWLEDGE = {
    "convection_diffusion": {
        "description": "Convection-diffusion: stabilized or DG (examples 07, 25, 42, 50)",
        "solver": "GMRES with diagonal preconditioner (non-symmetric system)",
        "elements": "ElementQuad1 (SUPG), ElementTriDG(ElementTriP1()) for DG",
        "pitfalls": [
            "Custom BilinearForm: access u.grad, v.grad for gradients",
            "For DG: use InteriorFacetBasis for jump terms",
            "Periodic mesh: example 42 shows advection on periodic domain",
            "High Peclet: use DG or increase mesh resolution",
        ],
    },
}

GENERATORS = {
    "convection_diffusion_2d": _convdiff_2d_skfem,
}
