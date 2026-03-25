"""scikit-fem Stokes flow generators and knowledge."""


def _stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Stokes flow with Taylor-Hood P2/P1."""
    nx = params.get("nx", 16)
    return f'''\
"""Stokes flow — Taylor-Hood P2/P1 — scikit-fem"""
from skfem import *
from skfem.models.poisson import vector_laplace, laplace, mass
import numpy as np
import json

m = MeshQuad.init_tensor(np.linspace(0, 1, {nx+1}), np.linspace(0, 1, {nx+1}))
e_u = ElementVector(ElementQuad2())
e_p = ElementQuad1()

ib_u = Basis(m, e_u)
ib_p = Basis(m, e_p)

@BilinearForm
def viscosity(u, v, w):
    return sum(u.grad[i].grad[j] * v.grad[i].grad[j]
               for i in range(2) for j in range(2))

# This is a simplified Stokes — for production use the block system
# K = [[A, B^T], [B, 0]] with divergence constraint

from skfem.models.general import divergence
K11 = asm(vector_laplace, ib_u)
K12 = -asm(divergence, ib_u, ib_p)

from scipy.sparse import bmat
K = bmat([[K11, K12], [K12.T, None]], format='csr')
f = np.zeros(K.shape[0])

# Velocity BCs — set for your problem
D_u = ib_u.get_dofs().flatten()
n_u = K11.shape[0]
# Set velocity BC values
d_vals = np.zeros(K.shape[0])

u = solve(*condense(K, f, D=np.concatenate([D_u, n_u + np.array([])]), x=d_vals))
print(f"Stokes: {{K.shape[0]}} DOFs")

summary = {{"n_dofs": K.shape[0]}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Stokes solve complete.")
'''


KNOWLEDGE = {
    "stokes": {
        "description": "Stokes flow — Taylor-Hood P2/P1 or Mini element (examples 18, 24, 30, 32)",
        "solver": "Block system: [[A, B^T], [B, 0]], solved via direct or Krylov-Uzawa",
        "elements": "Taylor-Hood: ElementVector(ElementTriP2()) + ElementTriP1(). Mini: ElementTriMini()",
        "pitfalls": [
            "System is indefinite — direct solve (UMFPACK) or Uzawa iteration",
            "Krylov-Uzawa (example 30): outer iteration on pressure Schur complement",
            "Must pin pressure at one point for enclosed flows. For open flows "
            "with do-nothing (traction-free) outlet BCs, pressure is uniquely "
            "determined without pinning.",
            "3D Stokes (example 32): uses ElementTetP2 + ElementTetP1",
            # DOF counting pitfall
            "CRITICAL: ib.Nbfun returns the per-ELEMENT DOF count, NOT the global "
            "DOF count. Use ib.N or the assembled matrix shape (A.shape[0]) for "
            "the global DOF count. Using Nbfun for DOF splitting silently produces "
            "wrong results with no error.",
            # ElementVector DOF ordering
            "ElementVector DOF ordering is unreliable for mixed systems. Safer "
            "approach: use two separate SCALAR P2 bases (one for u_x, one for u_y) "
            "and assemble an explicit block system. This is cleaner, debuggable, "
            "and avoids blocked-vs-interleaved ambiguity.",
            # Quadrature matching for mixed bases
            "When using asm() with two different bases (e.g., P2 velocity + P1 "
            "pressure), set intorder=4 (or higher) on BOTH bases to ensure matching "
            "quadrature. Without this, asm() raises 'Quadrature mismatch' errors.",
            # Pressure sign convention
            "scikit-fem Stokes uses -p*div(v) convention (same as FEniCS). NGSolve "
            "uses +p*div(v). Both are valid but produce different pressure signs. "
            "Be aware when comparing across solvers.",
        ],
    },
}

GENERATORS = {
    "stokes_2d": _stokes_2d,
}
