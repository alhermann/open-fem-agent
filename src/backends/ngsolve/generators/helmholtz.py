"""NGSolve Helmholtz equation generators and knowledge."""


def _helmholtz_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Helmholtz equation with PML absorbing layer."""
    k = params.get("k", 10)
    order = params.get("order", 4)
    maxh = params.get("maxh", 0.05)
    return f'''\
"""Helmholtz: -Δu - k²u = f with PML — NGSolve (complex-valued)"""
from ngsolve import *
from netgen.geom2d import SplineGeometry
import json

geo = SplineGeometry()
geo.AddCircle((0,0), r=1.0, bc="outer")
geo.AddCircle((0,0), r=0.7, leftdomain=2, rightdomain=1)
geo.SetMaterial(1, "pml")
geo.SetMaterial(2, "inner")
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))

mesh.SetPML(pml.Radial(rad=0.7, alpha=2j), definedon="pml")

fes = H1(mesh, order={order}, complex=True, dirichlet="outer")
u, v = fes.TnT()
k = {k}
a = BilinearForm(grad(u)*grad(v)*dx - k**2*u*v*dx).Assemble()

# Point source at origin
f = LinearForm(exp(-100*(x**2+y**2))*v*dx).Assemble()

gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

print(f"Helmholtz k={k}, DOFs: {{fes.ndof}}")
vtk = VTKOutput(mesh, coefs=[gfu.real, gfu.imag],
                names=["Re_u", "Im_u"], filename="result", subdivision=2)
vtk.Do()
summary = {{"k": k, "n_dofs": fes.ndof}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Helmholtz solve complete.")
'''


KNOWLEDGE = {
    "helmholtz": {
        "description": "Helmholtz equation with PML (perfectly matched layer)",
        "spaces": "H1(mesh, order=k, complex=True) — MUST use complex=True",
        "solver": "Direct for moderate k. For large k, use GMRES with multigrid",
        "pitfalls": [
            "complex=True flag required on FESpace",
            "PML: mesh.SetPML(pml.Radial(rad=r, alpha=a_j)) where alpha is imaginary",
            "PML alpha too small -> reflections; too large -> numerical instability",
            "Resolution: ~10 DOFs per wavelength (order p, h < lambda/(2p))",
            "For eigenvalues/resonances: ArnoldiSolver with shift-invert",
        ],
    },
}

GENERATORS = {
    "helmholtz_2d": _helmholtz_2d,
}
