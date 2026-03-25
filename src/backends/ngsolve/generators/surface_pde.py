"""NGSolve surface PDE generators and knowledge."""


def _surface_pde_3d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Laplace-Beltrami equation on a curved surface manifold (sphere)."""
    order = params.get("order", 3)
    maxh = params.get("maxh", 0.3)
    return f'''\
"""Laplace-Beltrami on sphere surface — surface FEM — NGSolve"""
from ngsolve import *
from netgen.occ import *
import json

# Geometry: sphere surface — set for your problem
sphere = Sphere(Pnt(0,0,0), r=1.0)
geo = OCCGeometry(sphere)
mesh = Mesh(geo.GenerateMesh(maxh={maxh}))
mesh.Curve({order})

print(f"Surface mesh: {{mesh.ne}} elements, {{mesh.nv}} vertices")

# H1 space on the surface manifold
fes = H1(mesh, order={order}, dirichlet="")
u, v = fes.TnT()

# Laplace-Beltrami: surface gradient on the manifold
# NGSolve automatically restricts grad to the tangent plane on surface meshes
a = BilinearForm(grad(u) * grad(v) * ds).Assemble()

# Source term on the surface — set for your problem
# Use spherical harmonics Y_2^0 as forcing: f = 6*z^2 - 2 (eigenfunction)
f_expr = 6 * z * z - 2
f = LinearForm(f_expr * v * ds).Assemble()

# Pin one DOF to fix the constant (Laplace-Beltrami has kernel = constants)
fes_constrained = H1(mesh, order={order})
u_c, v_c = fes_constrained.TnT()
a_c = BilinearForm(grad(u_c) * grad(v_c) * ds + 1e-8 * u_c * v_c * ds).Assemble()
f_c = LinearForm(f_expr * v_c * ds).Assemble()

gfu = GridFunction(fes_constrained)
gfu.vec.data = a_c.mat.Inverse(fes_constrained.FreeDofs()) * f_c.vec

# The exact solution is the spherical harmonic Y_2^0 = z^2 - 1/3
# (up to a constant shift)
max_val = max(gfu.vec)
min_val = min(gfu.vec)
print(f"Solution: max={{max_val:.8f}}, min={{min_val:.8f}}")

vtk = VTKOutput(mesh, coefs=[gfu], names=["solution"],
                filename="result", subdivision=2)
vtk.Do()

summary = {{
    "max_value": float(max_val),
    "min_value": float(min_val),
    "n_dofs": fes_constrained.ndof,
    "n_elements": mesh.ne,
    "order": {order},
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Surface PDE (Laplace-Beltrami) solve complete.")
'''


KNOWLEDGE = {
    "surface_pde": {
        "description": "PDEs on curved surface manifolds (Laplace-Beltrami, surface diffusion)",
        "spaces": "H1 on surface mesh (NGSolve automatically restricts to tangent plane)",
        "solver": "Direct or iterative — standard solvers work on surface meshes",
        "mesh": "OCC surfaces (Sphere, Cylinder, STEP import), mesh.Curve(order) for geometry approximation",
        "pitfalls": [
            "Use ds (surface measure) instead of dx (volume) for surface integrals",
            "grad on surface mesh automatically gives tangential (surface) gradient",
            "Laplace-Beltrami has kernel = constants; pin one DOF or add regularization",
            "mesh.Curve(order) improves geometry approximation for curved surfaces",
            "For evolving surfaces: use deformation mapping + ALE approach",
            "Surface meshes from OCC: Sphere, Cylinder, or any STEP/BREP surface",
        ],
    },
}

GENERATORS = {
    "surface_pde_3d": _surface_pde_3d,
}
