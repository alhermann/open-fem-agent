"""NGSolve Maxwell equations generators and knowledge."""


def _maxwell_3d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    3D magnetostatics with HCurl (Nedelec) elements."""
    order = params.get("order", 2)
    maxh = params.get("maxh", 0.3)
    return f'''\
"""Magnetostatics: curl-curl equation — HCurl (Nedelec) — NGSolve"""
from ngsolve import *
from netgen.csg import *
import json, math

# Geometry with material region — set for your problem
geo = CSGeometry()
outer = OrthoBrick(Pnt(-1,-1,-1), Pnt(1,1,1)).bc("outer")
inner = OrthoBrick(Pnt(-0.3,-0.3,-0.3), Pnt(0.3,0.3,0.3))
geo.Add(outer - inner)
geo.Add(inner, mat="source")
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))

fes = HCurl(mesh, order={order}, dirichlet="outer", nograds=True)
u, v = fes.TnT()

mu0 = 4*math.pi*1e-7
# curl-curl + regularization
a = BilinearForm(fes)
a += 1/mu0 * curl(u)*curl(v)*dx + 1e-8/mu0 * u*v*dx
a.Assemble()

# Source current — set for your problem
J = mesh.MaterialCF({{"source": (0, 0, 1)}}, default=(0, 0, 0))
f = LinearForm(J*v*dx).Assemble()

gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

# Magnetic field B = curl(A)
print(f"DOFs: {{fes.ndof}}, Elements: {{mesh.ne}}")
vtk = VTKOutput(mesh, coefs=[gfu, curl(gfu)], names=["A_field", "B_field"],
                filename="result", subdivision=0)
vtk.Do()
summary = {{"n_dofs": fes.ndof, "n_elements": mesh.ne}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Maxwell magnetostatics solve complete.")
'''


KNOWLEDGE = {
    "maxwell": {
        "description": "Maxwell equations with HCurl (Nedelec) edge elements",
        "spaces": "HCurl(mesh, order=k, nograds=True) — tangential continuity",
        "solver": "Direct for small. HCurlAMG preconditioner for large systems",
        "pitfalls": [
            "MUST use nograds=True to remove gradient kernel (otherwise singular)",
            "Add regularization: 1e-8*u*v*dx to make system invertible",
            "B = curl(A) — magnetic field is the curl of the vector potential",
            "Complex-valued for time-harmonic: HCurl(mesh, complex=True)",
            "3D only — 2D Maxwell reduces to scalar Helmholtz",
        ],
    },
}

GENERATORS = {
    "maxwell_3d_magnetostatics": _maxwell_3d,
}
