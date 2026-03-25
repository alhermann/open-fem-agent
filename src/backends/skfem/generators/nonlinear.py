"""scikit-fem nonlinear PDE generators and knowledge."""


def _nonlinear_2d_skfem(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Nonlinear PDE -div((1+u^2)*grad(u)) = f with manual Newton iteration."""
    nx = params.get("nx", 32)
    f_val = params.get("f", 1.0)
    tol = params.get("newton_tol", 1e-10)
    max_iter = params.get("max_iter", 50)
    return f'''\
"""Nonlinear PDE: -div((1+u^2)*grad(u)) = f — Newton iteration — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, mass, unit_load
import numpy as np
from scipy.sparse.linalg import spsolve
import json

# Mesh: structured quad mesh
m = MeshQuad.init_tensor(np.linspace(0, 1, {nx + 1}), np.linspace(0, 1, {nx + 1}))
e = ElementQuad1()
ib = Basis(m, e)

# Boundary DOFs
D = ib.get_dofs().flatten()
I = ib.complement_dofs(D)

# Nonlinear coefficient forms
@BilinearForm
def nonlinear_stiffness(u, v, w):
    # (1 + w_prev^2) * grad(u) . grad(v)
    u_prev = w["u_prev"]
    return (1 + u_prev ** 2) * (u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1])

@BilinearForm
def jacobian_extra(u, v, w):
    # 2 * w_prev * grad(w_prev) . grad(v) * u  (linearization of the nonlinear coefficient)
    u_prev = w["u_prev"]
    return 2 * u_prev * (w["u_prev_grad"][0] * v.grad[0] + w["u_prev_grad"][1] * v.grad[1]) * u

@LinearForm
def residual_form(v, w):
    # (1 + w_prev^2) * grad(w_prev) . grad(v) - f * v
    u_prev = w["u_prev"]
    return (1 + u_prev ** 2) * (w["u_prev_grad"][0] * v.grad[0] + w["u_prev_grad"][1] * v.grad[1]) - {f_val} * v

# Initial guess: zero
u = ib.zeros()

# Newton iteration
for it in range({max_iter}):
    # Interpolate current solution to quadrature points
    u_prev = ib.interpolate(u)
    u_prev_grad = u_prev.grad

    # Assemble residual
    R = residual_form.assemble(ib, u_prev=u_prev.value, u_prev_grad=u_prev_grad)

    # Assemble Jacobian: d/du [(1+u^2)*grad(u).grad(v)]
    J1 = nonlinear_stiffness.assemble(ib, u_prev=u_prev.value)
    J2 = jacobian_extra.assemble(ib, u_prev=u_prev.value, u_prev_grad=u_prev_grad)
    J = J1 + J2

    # Apply boundary conditions
    R[D] = 0.0
    du = np.zeros_like(u)
    du[I] = spsolve(J[I][:, I], -R[I])

    u += du
    res_norm = np.linalg.norm(du[I])
    print(f"Newton it {{it+1}}: ||du|| = {{res_norm:.6e}}")
    if res_norm < {tol}:
        print(f"Converged in {{it+1}} iterations")
        break

max_val = u.max()
print(f"max(u) = {{max_val:.10f}}")
print(f"DOFs: {{len(u)}}")

import meshio
cells = [("quad", m.t.T)]
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])]) if m.p.shape[0] == 2 else m.p.T
mio = meshio.Mesh(points, cells, point_data={{"phi": u}})
mio.write("result.vtu")

summary = {{
    "max_value": float(max_val),
    "n_dofs": len(u),
    "n_elements": m.nelements,
    "newton_iterations": it + 1,
    "element_type": "Q1 quad",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Nonlinear PDE solve complete.")
'''


KNOWLEDGE = {
    "nonlinear": {
        "description": "Nonlinear PDE via manual Newton iteration (scikit-fem)",
        "solver": "Manual Newton loop: assemble Jacobian + residual, solve with spsolve",
        "elements": "ElementQuad1, ElementTriP1 (any standard H1 element)",
        "pitfalls": [
            "scikit-fem does NOT have a built-in Newton solver — manual loop required",
            "Jacobian: linearize the nonlinear weak form w.r.t. solution and assemble",
            "Use ib.interpolate(u) to evaluate solution at quadrature points",
            "ib.interpolate(u).grad gives gradient at quadrature points",
            "Convergence: quadratic near solution if Jacobian is exact",
            "For difficult problems: add line search or damping (u += alpha * du)",
        ],
    },
}

GENERATORS = {
    "nonlinear_2d": _nonlinear_2d_skfem,
}
