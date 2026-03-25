"""NGSolve Stokes flow generators and knowledge."""


def _stokes_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Stokes flow with Taylor-Hood P2/P1 elements."""
    nx = params.get("nx", 32)
    nu_visc = params.get("viscosity", 1.0)
    maxh = 1.0 / nx
    return f'''\
"""Stokes flow — Taylor-Hood P2/P1 — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))

V = VectorH1(mesh, order=2, dirichlet="bottom|right|top|left")
Q = H1(mesh, order=1)
X = V * Q
(u, p), (v, q) = X.TnT()

nu = {nu_visc}
a = BilinearForm(X)
a += nu * InnerProduct(Grad(u), Grad(v)) * dx
a += div(u)*q*dx + div(v)*p*dx
a.Assemble()

f = LinearForm(X)
f.Assemble()

gfu = GridFunction(X)
# Velocity BC — set for your problem
uin = CoefficientFunction((1, 0))
gfu.components[0].Set(uin, definedon=mesh.Boundaries("top"))

# Solve with modified RHS for non-homogeneous Dirichlet
f.vec.data -= a.mat * gfu.vec
# Try available direct solvers (umfpack may not be installed)
inv = None
for solver_name in ["pardiso", "mumps", "umfpack"]:
    try:
        inv = a.mat.Inverse(X.FreeDofs(), solver_name)
        break
    except:
        pass
if inv is None:
    from ngsolve.krylovspace import MinResSolver
    inv = MinResSolver(a.mat, freedofs=X.FreeDofs(), maxsteps=10000, tol=1e-10)
gfu.vec.data += inv * f.vec

vel = gfu.components[0]
pres = gfu.components[1]
max_vel = Integrate(InnerProduct(vel, vel), mesh)
print(f"L2(velocity) = {{max_vel**0.5:.6f}}")

vtk = VTKOutput(mesh, coefs=[vel, pres], names=["velocity", "pressure"],
                filename="result", subdivision=1)
vtk.Do()
summary = {{"l2_velocity": float(max_vel**0.5), "n_dofs": X.ndof}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Stokes solve complete.")
'''


KNOWLEDGE = {
    "stokes": {
        "description": "Stokes flow with Taylor-Hood P2/P1 or Mini element or HDG",
        "spaces": "VectorH1(order=2) * H1(order=1) for Taylor-Hood. VectorH1 * L2 for DG-Stokes",
        "solver": "Direct: pardiso > mumps > umfpack (try in order). Iterative: MinRes or GMRES (indefinite system!)",
        "pitfalls": [
            "Stokes system is INDEFINITE — use MinRes or GMRES, never CG",
            "Compound space: (u,p),(v,q) = X.TnT() — returns nested tuples",
            "Pressure has no Dirichlet BC for enclosed flows — use mean-free "
            "condition via NumberSpace or pinning one node. For open flows with "
            "do-nothing outlet, pressure is uniquely determined.",
            "Block preconditioners: BlockMatrix + inverse of Schur complement approx",
            # Direct solver availability
            "Do NOT hardcode 'umfpack' — it may not be available. Use a fallback "
            "pattern: for name in ['pardiso','mumps','umfpack']: try Inverse(..., name)",
            # Pressure sign convention
            "NGSolve Stokes uses +p*div(v) convention (opposite sign from FEniCS). "
            "Both are valid. Be aware when comparing pressure fields across solvers.",
        ],
    },
}

GENERATORS = {
    "stokes_2d": _stokes_2d,
    "stokes_2d_hdg": _stokes_2d,  # Same solver, HDG variant TBD
}
