"""NGSolve eigenvalue problem generators and knowledge."""


def _eigenvalue_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Eigenvalue problem: Laplace on unit square."""
    order = params.get("order", 4)
    n_eigs = params.get("n_eigenvalues", 10)
    maxh = params.get("maxh", 0.03)
    return f'''\
"""Eigenvalue problem: Laplace — ArnoldiSolver — NGSolve"""
from ngsolve import *
import json, math

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))
fes = H1(mesh, order={order}, dirichlet="bottom|right|top|left")
u, v = fes.TnT()

a = BilinearForm(grad(u)*grad(v)*dx).Assemble()
m = BilinearForm(u*v*dx).Assemble()

gfu = GridFunction(fes, multidim={n_eigs})
lam = ArnoldiSolver(a.mat, m.mat, fes.FreeDofs(), list(gfu.vecs), shift=0)

print(f"First {n_eigs} eigenvalues:")
exact = [math.pi**2*(i**2+j**2) for i in range(1,6) for j in range(1,6)]
exact.sort()
for i, (computed, ref) in enumerate(zip(lam, exact[:{n_eigs}])):
    err = abs(computed - ref) / ref
    print(f"  lambda_{{i+1}} = {{computed:.6f}} (exact: {{ref:.6f}}, error: {{err:.2e}})")

vtk = VTKOutput(mesh, coefs=[gfu.components[0]], names=["eigenmode_1"],
                filename="result", subdivision=1)
vtk.Do()
summary = {{"eigenvalues": [float(l) for l in lam], "n_dofs": fes.ndof}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Eigenvalue solve complete.")
'''


KNOWLEDGE = {
    "eigenvalue": {
        "description": "Eigenvalue problems via ArnoldiSolver (shift-invert Arnoldi)",
        "spaces": "Any H1 space",
        "solver": "ArnoldiSolver(a.mat, m.mat, freedofs, vecs, shift=target)",
        "pitfalls": [
            "shift parameter targets eigenvalues near that value",
            "GridFunction(fes, multidim=n) allocates space for n eigenvectors",
            "Exact eigenvalues of Laplace on [0,1]^2: pi^2(m^2+n^2)",
            "For generalized eigenvalue: A*x = lambda*M*x",
            "Alternative: PINVIT (preconditioned inverse iteration) for lowest eigenvalues",
        ],
    },
}

GENERATORS = {
    "eigenvalue_2d": _eigenvalue_2d,
}
