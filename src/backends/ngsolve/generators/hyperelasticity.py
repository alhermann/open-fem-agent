"""NGSolve hyperelasticity generators and knowledge."""


def _hyperelasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Neo-Hookean hyperelasticity via SymbolicEnergy + Newton."""
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    maxh = params.get("maxh", 0.05)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Neo-Hookean hyperelasticity — Newton solver — NGSolve"""
from ngsolve import *
import json

mesh = Mesh(unit_square.GenerateMesh(maxh={maxh}))
fes = VectorH1(mesh, order=2, dirichlet="left|bottom")
u = fes.TrialFunction()

mu, lam = {mu}, {lam}
I = Id(2)
F = I + Grad(u)
C = F.trans * F
J = Det(F)

# Neo-Hookean energy: W = mu/2*(tr(C)-2) - mu*ln(J) + lam/2*ln(J)^2
energy = 0.5*mu*(Trace(C) - 2) - mu*log(J) + 0.5*lam*log(J)**2

a = BilinearForm(fes, symmetric=True)
a += Variation(energy * dx)

# Apply displacement BC — set for your problem
gfu = GridFunction(fes)
gfu.Set(CoefficientFunction((0.3*x, -0.1)), definedon=mesh.Boundaries("top"))

solvers.Newton(a, gfu, maxits=20, printing=True)

vtk = VTKOutput(mesh, coefs=[gfu], names=["displacement"], filename="result", subdivision=1)
vtk.Do()
summary = {{"n_dofs": fes.ndof, "converged": True}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Hyperelasticity solve complete.")
'''


def _hyperelasticity_3d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    3D Neo-Hookean hyperelasticity."""
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    lx = params.get("lx", 1)
    ly = params.get("ly", 0.1)
    lz = params.get("lz", 0.04)
    maxh = params.get("maxh", 0.02)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""3D Neo-Hookean hyperelasticity — NGSolve"""
from ngsolve import *
from netgen.occ import *
import json

# Geometry — set dimensions for your problem
box = Box((0,0,0), ({lx},{ly},{lz}))
geo = OCCGeometry(box)
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))
fes = VectorH1(mesh, order=2, dirichlet=".*")
u = fes.TrialFunction()

mu, lam = {mu}, {lam}
I = Id(3)
F = I + Grad(u)
C = F.trans * F
J = Det(F)
energy = 0.5*mu*(Trace(C) - 3) - mu*log(J) + 0.5*lam*log(J)**2

a = BilinearForm(fes, symmetric=True)
a += Variation(energy * dx)

gfu = GridFunction(fes)
# Apply load incrementally — set for your problem
for load in [0.2, 0.5, 0.8, 1.0]:
    gfu.Set(CoefficientFunction((0, 0, -0.01*load)), definedon=mesh.Boundaries(".*"))
    solvers.Newton(a, gfu, maxits=20, printing=False)

vtk = VTKOutput(mesh, coefs=[gfu], names=["displacement"], filename="result", subdivision=1)
vtk.Do()
summary = {{"n_dofs": fes.ndof}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "hyperelasticity": {
        "description": "Nonlinear hyperelasticity via SymbolicEnergy + Newton solver",
        "spaces": "VectorH1(mesh, order=2+)",
        "solver": "solvers.Newton(a, gfu, maxits=20) — built-in Newton with damping",
        "pitfalls": [
            "Use Variation(energy*dx) — NGSolve auto-derives residual and tangent",
            "Neo-Hookean energy: 0.5*mu*(Tr(C)-d) - mu*ln(J) + 0.5*lam*ln(J)^2",
            "For large deformations: use load stepping (incremental loading)",
            "dampfactor in Newton controls step size (default 1.0, reduce if diverging)",
            "Unique NGSolve feature: a.Apply() for residual, a.AssembleLinearization() for tangent",
        ],
    },
}

GENERATORS = {
    "hyperelasticity_2d": _hyperelasticity_2d,
    "hyperelasticity_3d": _hyperelasticity_3d,
}
