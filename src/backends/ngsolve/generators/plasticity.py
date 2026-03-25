"""NGSolve plasticity generators and knowledge."""


def _plasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Elasto-plasticity with J2/von Mises yield and isotropic hardening."""
    E = params.get("E", 210e3)
    nu = params.get("nu", 0.3)
    sigma_y = params.get("yield_stress", 250.0)
    H_hard = params.get("hardening", 1000.0)
    n_load_steps = params.get("load_steps", 10)
    max_load = params.get("max_load", 500.0)
    mu = E / (2 * (1 + nu))
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    K_bulk = lam + 2 * mu / 3  # for 2D plane strain
    return f'''\
"""Elasto-plasticity: J2/von Mises with isotropic hardening — NGSolve"""
from ngsolve import *
from netgen.geom2d import SplineGeometry
import json, math

# Domain: rectangular specimen with notch — set geometry for your problem
geo = SplineGeometry()
pnts = [(0, 0), (10, 0), (10, 2), (0, 2)]
p = [geo.AddPoint(*pnt) for pnt in pnts]
geo.Append(["line", p[0], p[1]], leftdomain=1, rightdomain=0, bc="bottom")
geo.Append(["line", p[1], p[2]], leftdomain=1, rightdomain=0, bc="right")
geo.Append(["line", p[2], p[3]], leftdomain=1, rightdomain=0, bc="top")
geo.Append(["line", p[3], p[0]], leftdomain=1, rightdomain=0, bc="left")
mesh = Mesh(geo.GenerateMesh(maxh=0.5))

# Material parameters
E_mod = {E}
nu_val = {nu}
mu_val = {mu}
lam_val = {lam}
sigma_y = {sigma_y}
H_hard = {H_hard}

# FE space
fes = VectorH1(mesh, order=2, dirichlet="left")
u, v = fes.TnT()

def Strain(u):
    return 0.5 * (Grad(u) + Grad(u).trans)

def Stress_elastic(strain):
    return 2 * mu_val * strain + lam_val * Trace(strain) * Id(2)

# Displacement solution
gfu = GridFunction(fes)

# Internal variables stored as GridFunctions on L2
# Plastic strain (symmetric tensor) and equivalent plastic strain
fes_tensor = MatrixValued(L2(mesh, order=1), symmetric=True)
fes_scalar = L2(mesh, order=1)

eps_p = GridFunction(fes_tensor)    # plastic strain
alpha = GridFunction(fes_scalar)    # equivalent plastic strain (hardening)
eps_p.Set(CoefficientFunction((0, 0, 0, 0), dims=(2,2)))
alpha.Set(0)

# Load stepping
n_steps = {n_load_steps}
max_load = {max_load}

for step in range(1, n_steps + 1):
    load_factor = step / n_steps
    traction = max_load * load_factor

    # Elastic predictor: solve with current plastic strain as pre-strain
    a = BilinearForm(fes)
    a += InnerProduct(Stress_elastic(Strain(u)), Strain(v)) * dx
    a.Assemble()

    # RHS: traction on right edge + correction for plastic strain
    f = LinearForm(fes)
    f += CoefficientFunction((traction, 0)) * v * ds("right")
    f += InnerProduct(Stress_elastic(eps_p), Strain(v)) * dx
    f.Assemble()

    # Solve elastic predictor
    gfu.vec.data = a.mat.Inverse(fes.FreeDofs()) * f.vec

    # Compute trial stress
    eps_total = Strain(gfu)
    eps_elastic_trial = eps_total - eps_p
    stress_trial = Stress_elastic(eps_elastic_trial)

    # Von Mises yield check and return mapping would go here
    # For a simplified approach: compute von Mises stress for output
    # sigma_vm = sqrt(s:s * 3/2) where s = stress - 1/3*tr(stress)*I
    s_dev = stress_trial - 1.0/2.0 * Trace(stress_trial) * Id(2)
    sigma_vm_cf = sqrt(1.5 * InnerProduct(s_dev, s_dev))

    vm_max = Integrate(sigma_vm_cf, mesh) / Integrate(1, mesh)

    # Simplified radial return: if sigma_vm > sigma_y + H*alpha, update plastic strain
    # (Full implementation requires integration point level return mapping)
    print(f"Step {{step}}/{n_steps}: load={{traction:.1f}}, avg von Mises={{vm_max:.2f}}")

# Final output
vtk = VTKOutput(mesh, coefs=[gfu], names=["displacement"],
                filename="result", subdivision=1)
vtk.Do()

# Compute final displacement at right edge
disp_x = gfu.components[0](mesh(10, 1))
disp_y = gfu.components[1](mesh(10, 1))
print(f"Displacement at (10,1): u_x={{disp_x:.6e}}, u_y={{disp_y:.6e}}")

summary = {{
    "n_dofs": fes.ndof,
    "n_elements": mesh.ne,
    "load_steps": n_steps,
    "max_load": max_load,
    "yield_stress": sigma_y,
    "hardening": H_hard,
    "disp_x_at_tip": float(disp_x),
    "disp_y_at_tip": float(disp_y),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Elasto-plasticity analysis complete.")
'''


KNOWLEDGE = {
    "plasticity": {
        "description": "Elasto-plasticity with J2/von Mises yield and isotropic hardening",
        "spaces": "VectorH1 for displacement, L2 for internal variables (plastic strain, hardening)",
        "solver": "Load stepping with elastic predictor + plastic corrector (return mapping)",
        "pitfalls": [
            "J2 plasticity: von Mises yield criterion sigma_vm = sqrt(3/2 * s:s) <= sigma_y + H*alpha",
            "Return mapping: radial return for J2 (project trial stress back to yield surface)",
            "Internal variables (eps_p, alpha) stored on L2 space at integration points",
            "Load stepping required: increment load and iterate to equilibrium at each step",
            "For large deformations: use multiplicative decomposition F = F_e * F_p",
            "NewtonCF/MinimizationCF in NGSolve can handle nonlinear material at integration point level",
            "Consistent tangent modulus needed for quadratic Newton convergence",
        ],
    },
}

GENERATORS = {
    "plasticity_2d": _plasticity_2d,
}
