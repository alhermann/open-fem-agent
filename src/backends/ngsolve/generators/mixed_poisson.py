"""NGSolve mixed Poisson generators and knowledge."""


def _mixed_poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Mixed Poisson with H(div)/L2 (Raviart-Thomas + piecewise constants)."""
    order = params.get("order", 1)
    return f'''\
"""Mixed Poisson: sigma = -grad(u), div(sigma) = f — HDiv/L2 — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh=0.05))

# Raviart-Thomas for flux, L2 for scalar
V = HDiv(mesh, order={order}, RT=True)
Q = L2(mesh, order={order}-1)
X = V * Q
(sigma, u), (tau, v) = X.TnT()

n = specialcf.normal(2)
a = BilinearForm(X)
a += (sigma*tau + div(sigma)*v + div(tau)*u)*dx
a.Assemble()

f = LinearForm(X)
f += -1*v*dx  # source f=1
f.Assemble()

gfu = GridFunction(X)
gfu.vec.data = a.mat.Inverse(X.FreeDofs()) * f.vec

flux = gfu.components[0]
scalar = gfu.components[1]
max_u = Integrate(scalar, mesh) / Integrate(1, mesh)  # mean value
print(f"Mean u: {{max_u:.6f}}")

vtk = VTKOutput(mesh, coefs=[flux, scalar], names=["flux", "potential"],
                filename="result", subdivision=0)
vtk.Do()
summary = {{"n_dofs": X.ndof, "mean_value": float(max_u)}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Mixed Poisson solve complete.")
'''


KNOWLEDGE = {
    "mixed_poisson": {
        "description": "Mixed Poisson with H(div)/L2 (Raviart-Thomas flux recovery)",
        "spaces": "HDiv(mesh, order=k, RT=True) * L2(mesh, order=k-1)",
        "solver": "Direct (saddle-point) or iterative with Schur complement",
        "pitfalls": [
            "RT=True for Raviart-Thomas, RT=False (default) for BDM elements",
            "Normal component continuous across elements; div well-defined",
            "Saddle-point system — use direct solver or block preconditioner",
        ],
    },
}

GENERATORS = {
    "mixed_poisson_2d": _mixed_poisson_2d,
}
