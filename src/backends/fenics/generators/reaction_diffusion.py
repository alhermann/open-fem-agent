"""Reaction-diffusion system generator for FEniCSx/dolfinx.

Variants: 2d
"""


KNOWLEDGE = {
    "description": "Two-species reaction-diffusion system (coupled, transient, nonlinear)",
    "weak_form": "((u-u_old)/dt, phi_u)*dx + D1*(grad(u),grad(phi_u))*dx = (R_u, phi_u)*dx (+ same for v)",
    "function_space": "Mixed: P1 + P1 (one per species) via mixed_element",
    "solver": "Newton (PETSc SNES) with LU direct for each time step",
    "pitfalls": [
        "Nonlinear reaction terms require Newton iteration (NonlinearProblem)",
        "Time stepping: backward Euler is robust; Crank-Nicolson can oscillate",
        "Initial conditions: must set on collapsed sub-spaces, then scatter_forward",
        "No-flux (Neumann zero) is the natural BC \u2014 no Dirichlet needed for insulated walls",
        "For stiff reactions: may need smaller dt or implicit-explicit (IMEX) splitting",
        "Common reaction models: Gray-Scott, Schnakenberg, Brusselator, Lotka-Volterra",
        "Conservation: check mass integrals over time to verify correctness",
    ],
    "materials": {
        "D1": {"range": [1e-6, 10.0], "unit": "m^2/s (diffusion coefficient, species 1)"},
        "D2": {"range": [1e-6, 10.0], "unit": "m^2/s (diffusion coefficient, species 2)"},
    },
}

VARIANTS = ["2d"]


def generate(variant: str, params: dict) -> str:
    """Dispatch to the appropriate reaction-diffusion variant."""
    generators = {
        "2d": _reaction_diffusion_2d,
    }
    gen = generators.get(variant)
    if not gen:
        raise ValueError(f"Unknown reaction_diffusion variant: {variant!r}. Available: {list(generators)}")
    return gen(params)


def _reaction_diffusion_2d(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable script. All parameter defaults are placeholders. The user/agent must set values appropriate to the specific problem being solved."""
    nx = params.get("nx", 64)
    ny = params.get("ny", 64)
    D1 = params.get("D1", 0.01)
    D2 = params.get("D2", 0.005)
    n_steps = params.get("n_steps", 100)
    dt = params.get("dt", 0.01)
    return f'''\
"""Two-species reaction-diffusion system — FEniCSx/dolfinx
du/dt = D1 * laplacian(u) + R_u(u, v)
dv/dt = D2 * laplacian(v) + R_v(u, v)
Backward Euler time stepping, coupled system.
"""
from mpi4py import MPI
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
import ufl
import numpy as np
from basix.ufl import element, mixed_element

# Mesh
domain = mesh.create_unit_square(MPI.COMM_WORLD, {nx}, {ny}, mesh.CellType.triangle)
tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

# Mixed function space for two species
P1 = element("Lagrange", domain.topology.cell_name(), 1)
ME = mixed_element([P1, P1])
W = fem.functionspace(domain, ME)

# Previous time step solution
w_old = fem.Function(W)
(u_old, v_old) = ufl.split(w_old)

# Current time step solution
w = fem.Function(W)
(u, v) = ufl.split(w)

# Test functions
(phi_u, phi_v) = ufl.TestFunctions(W)

# Parameters
D1 = fem.Constant(domain, default_scalar_type({D1}))
D2 = fem.Constant(domain, default_scalar_type({D2}))
dt = fem.Constant(domain, default_scalar_type({dt}))

# Reaction terms — generic reaction kinetics (user should customize)
# Example: Gray-Scott-type or Lotka-Volterra-type
r_u = u * (1.0 - u) - u * v
r_v = -v + u * v

# Weak form (backward Euler)
F = ((u - u_old) / dt * phi_u + D1 * ufl.dot(ufl.grad(u), ufl.grad(phi_u)) - r_u * phi_u) * ufl.dx \\
  + ((v - v_old) / dt * phi_v + D2 * ufl.dot(ufl.grad(v), ufl.grad(phi_v)) - r_v * phi_v) * ufl.dx

# No-flux (Neumann zero) BCs — natural, no Dirichlet needed

# Initial conditions — localized perturbation
def init_u(x):
    return 0.5 + 0.1 * np.exp(-50 * ((x[0] - 0.5)**2 + (x[1] - 0.5)**2))

def init_v(x):
    return 0.25 + 0.1 * np.exp(-50 * ((x[0] - 0.3)**2 + (x[1] - 0.3)**2))

V0, _ = W.sub(0).collapse()
V1, _ = W.sub(1).collapse()
u_init = fem.Function(V0)
u_init.interpolate(init_u)
v_init = fem.Function(V1)
v_init.interpolate(init_v)
w.sub(0).interpolate(u_init)
w.sub(1).interpolate(v_init)
w.x.scatter_forward()
w_old.x.array[:] = w.x.array[:]

# Newton solver
problem = NonlinearProblem(F, w, bcs=[], petsc_options_prefix="rd",
    petsc_options={{"ksp_type": "preonly", "pc_type": "lu",
                   "pc_factor_mat_solver_type": "mumps",
                   "snes_rtol": 1e-8, "snes_max_it": 30}})

# Time loop
n_steps = {n_steps}
for step in range(n_steps):
    w_old.x.array[:] = w.x.array[:]
    problem.solve()
    if step % max(1, n_steps // 10) == 0:
        (u_s, v_s) = w.split()
        print(f"Step {{step}}: u in [{{w.sub(0).collapse().x.array.min():.4f}}, {{w.sub(0).collapse().x.array.max():.4f}}]")

# Output final state
u_out = w.sub(0).collapse()
v_out = w.sub(1).collapse()
u_out.name = "species_u"
v_out.name = "species_v"

# Interpolate to P1 for output
V_out = fem.functionspace(domain, ("Lagrange", 1))
u_p1 = fem.Function(V_out, name="species_u")
v_p1 = fem.Function(V_out, name="species_v")
u_p1.interpolate(u_out)
v_p1.interpolate(v_out)

from dolfinx.io import XDMFFile
with XDMFFile(domain.comm, "species_u.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_p1)
with XDMFFile(domain.comm, "species_v.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(v_p1)

print(f"Reaction-diffusion: {{n_steps}} steps complete")
print(f"u: [{{u_p1.x.array.min():.6e}}, {{u_p1.x.array.max():.6e}}]")
print(f"v: [{{v_p1.x.array.min():.6e}}, {{v_p1.x.array.max():.6e}}]")
print(f"DOFs: {{W.dofmap.index_map.size_global}}")
'''
