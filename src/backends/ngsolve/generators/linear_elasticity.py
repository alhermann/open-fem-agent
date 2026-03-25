"""NGSolve linear elasticity generators and knowledge."""


def _elasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Linear elasticity on rectangular domain, fixed left edge, body force."""
    nx = params.get("nx", 40)
    ny = params.get("ny", 4)
    lx = params.get("lx", 10.0)
    ly = params.get("ly", 1.0)
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    maxh = lx / nx
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""Linear elasticity: rectangular domain, fixed left, body force — NGSolve"""
from ngsolve import *
import json
from netgen.geom2d import SplineGeometry

# Domain mesh — set geometry for your problem
geo = SplineGeometry()
pnts = [(0, 0), ({lx}, 0), ({lx}, {ly}), (0, {ly})]
p = [geo.AddPoint(*pnt) for pnt in pnts]
geo.Append(["line", p[0], p[1]], leftdomain=1, rightdomain=0, bc="bottom")
geo.Append(["line", p[1], p[2]], leftdomain=1, rightdomain=0, bc="right")
geo.Append(["line", p[2], p[3]], leftdomain=1, rightdomain=0, bc="top")
geo.Append(["line", p[3], p[0]], leftdomain=1, rightdomain=0, bc="left")
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))

# Vector FE space
fes = VectorH1(mesh, order=1, dirichlet="left")
u, v = fes.TnT()

mu_val = {mu}
lam_val = {lam}

def Strain(u):
    return 0.5 * (Grad(u) + Grad(u).trans)

def Stress(u):
    return 2 * mu_val * Strain(u) + lam_val * Trace(Strain(u)) * Id(2)

a = BilinearForm(InnerProduct(Stress(u), Strain(v)) * dx).Assemble()
f = LinearForm(CoefficientFunction((0, -1)) * v * dx).Assemble()

gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

# Max tip displacement
disp = gfu.components
max_uy = 0
for p in mesh.vertices:
    val = disp[1](mesh(*p.point))
    if abs(val) > abs(max_uy):
        max_uy = val

print(f"Max tip displacement (y): {{max_uy:.6f}}")

vtk = VTKOutput(mesh, coefs=[gfu], names=["displacement"],
                filename="result", subdivision=0)
vtk.Do()

summary = {{
    "max_displacement_y": float(max_uy),
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Elasticity solve complete.")
'''


def _elasticity_3d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    3D linear elasticity with OCC geometry."""
    E = params.get("E", 1000.0)
    nu = params.get("nu", 0.3)
    lx = params.get("lx", 10)
    ly = params.get("ly", 1)
    lz = params.get("lz", 1)
    maxh = params.get("maxh", 0.5)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    return f'''\
"""3D Linear elasticity — NGSolve + OCC geometry"""
from ngsolve import *
from netgen.occ import *
import json

# Geometry — set dimensions for your problem
box = Box((0,0,0), ({lx},{ly},{lz}))
geo = OCCGeometry(box)
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))

fes = VectorH1(mesh, order=2, dirichlet=".*")
# Fix left face only
fes = VectorH1(mesh, order=2, dirichlet="face3")  # face numbering depends on OCC
u, v = fes.TnT()

mu_val, lam_val = {mu}, {lam}
def Strain(u): return 0.5*(Grad(u) + Grad(u).trans)
def Stress(u): return 2*mu_val*Strain(u) + lam_val*Trace(Strain(u))*Id(3)

a = BilinearForm(InnerProduct(Stress(u), Strain(v))*dx).Assemble()
f = LinearForm(CoefficientFunction((0, -1, 0))*v*dx).Assemble()

gfu = GridFunction(fes)
gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec
print(f"DOFs: {{fes.ndof}}")

vtk = VTKOutput(mesh, coefs=[gfu], names=["displacement"], filename="result", subdivision=1)
vtk.Do()
summary = {{"n_dofs": fes.ndof, "n_elements": mesh.ne}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "linear_elasticity": {
        "description": "Linear elasticity (plane strain/stress, 3D) with VectorH1",
        "spaces": "VectorH1 (NOT H1 with dim parameter — that creates CompoundFESpace)",
        "solver": "Direct for small, CG+AMG for large. Preconditioners: bddc, multigrid",
        "pitfalls": [
            "Use VectorH1(mesh, order=2, dirichlet='fix'), NOT H1(mesh, dim=2)",
            "Body forces: CoefficientFunction((fx, fy)) for 2D, (fx, fy, fz) for 3D",
            "Evaluation: gfu.components[i](mesh(x,y)) for component-wise point values",
            "Stress tensor: use MatrixValued(H1(mesh, order=k), symmetric=True) for visualization",
            "Plane strain: use standard Lame parameters. Plane stress: modify lambda",
        ],
    },
}

GENERATORS = {
    "linear_elasticity_2d": _elasticity_2d,
    "linear_elasticity_3d": _elasticity_3d,
}
