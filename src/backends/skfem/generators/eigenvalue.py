"""scikit-fem eigenvalue problem generators and knowledge."""


def _eigenvalue_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Eigenvalue problem: Laplace on unit square."""
    nx = params.get("nx", 32)
    n_eigs = params.get("n_eigenvalues", 5)
    return f'''\
"""Eigenvalue problem: Laplace eigenvalues — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, mass
import numpy as np
from scipy.sparse.linalg import eigsh
import json, math

m = MeshQuad.init_tensor(np.linspace(0, 1, {nx+1}), np.linspace(0, 1, {nx+1}))
e = ElementQuad1()
ib = Basis(m, e)

K = laplace.assemble(ib)
M = mass.assemble(ib)

D = ib.get_dofs().flatten()
I = ib.complement_dofs(D)

# Solve generalized eigenvalue: K*x = lambda*M*x (restrict to interior DOFs)
eigenvalues, eigenvectors = eigsh(K[I][:, I], k={n_eigs}, M=M[I][:, I], sigma=0, which='LM')

# Exact eigenvalues: pi^2*(m^2+n^2)
exact = sorted([math.pi**2*(i**2+j**2) for i in range(1,6) for j in range(1,6)])[:{n_eigs}]

print(f"Computed eigenvalues: {{eigenvalues}}")
print(f"Exact eigenvalues:    {{exact}}")
for i, (c, e_val) in enumerate(zip(eigenvalues, exact)):
    err = abs(c - e_val) / e_val
    print(f"  lambda_{{i+1}} = {{c:.6f}} (exact: {{e_val:.6f}}, err: {{err:.2e}})")

summary = {{"eigenvalues": eigenvalues.tolist(), "exact": exact, "n_dofs": K.shape[0]}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Eigenvalue solve complete.")
'''


KNOWLEDGE = {
    "eigenvalue": {
        "description": "Eigenvalue problems — Laplace, elasticity vibration (examples 02, 03, 16, 21)",
        "solver": "scipy.sparse.linalg.eigsh (Lanczos for symmetric generalized eigenvalue)",
        "pitfalls": [
            "Restrict to interior DOFs: K[I][:,I], M[I][:,I] where I = complement_dofs(D)",
            "sigma=0 for shift-invert (finds smallest eigenvalues)",
            "Exact Laplace eigenvalues on [0,1]^2: pi^2(m^2+n^2)",
            "For structural vibration: K*x = omega^2*M*x",
        ],
    },
}

GENERATORS = {
    "eigenvalue_2d": _eigenvalue_2d,
}
