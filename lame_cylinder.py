"""
3D Thick-Walled Cylinder under Internal Pressure
Neo-Hookean Hyperelasticity vs Analytical Lamé Solution (Plane Strain)

Quarter-cylinder with symmetry BCs, Gmsh mesh, FEniCSx solver.

References:
  - Timoshenko & Goodier, Theory of Elasticity (Lamé solution)
  - Solids4foam pressurised cylinder tutorial
  - FEniCSx hyperelasticity tutorial (jsdokken.com)

Parameters from literature:
  a=1, b=2, E=1000, nu=0.3, p_i=5 => ~1% max strain (small-strain regime)
"""

import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import gmsh
from dolfinx import mesh, fem, io, default_scalar_type
from dolfinx.fem.petsc import NonlinearProblem
from dolfinx.io import gmshio
import ufl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ============================================================
# 1. Parameters
# ============================================================
a_r = 1.0       # inner radius
b_r = 2.0       # outer radius
L = 1.0         # axial length
E_val = 1000.0  # Young's modulus
nu_val = 0.3    # Poisson's ratio
p_i = 5.0       # internal pressure
h = 0.1         # mesh element size

# Derived Neo-Hookean (Lamé) parameters
mu_val = E_val / (2.0 * (1.0 + nu_val))           # = 384.615
lmbda_val = E_val * nu_val / ((1.0 + nu_val) * (1.0 - 2.0 * nu_val))  # = 576.923

print("=" * 60)
print("THICK-WALLED CYLINDER — Neo-Hookean vs Lamé")
print("=" * 60)
print(f"Geometry: a = {a_r}, b = {b_r}, L = {L}")
print(f"Material: E = {E_val}, nu = {nu_val}")
print(f"Neo-Hookean: mu = {mu_val:.3f}, lambda = {lmbda_val:.3f}")
print(f"Loading: p_i = {p_i}")
print(f"Mesh size: h = {h}")

# ============================================================
# 2. Analytical Lamé solution (plane strain)
# ============================================================
# u_r(r) = (1+nu)*p*a^2 / [E*(b^2-a^2)] * [(1-2nu)*r + b^2/r]
# sigma_r(r) = p*a^2/(b^2-a^2) * (1 - b^2/r^2)
# sigma_theta(r) = p*a^2/(b^2-a^2) * (1 + b^2/r^2)

def lame_ur(r):
    """Radial displacement — plane strain Lamé solution (corrected).
    Derived from u_r = A*r + B/r with sigma_r(a)=-p, sigma_r(b)=0.
    The (1+nu) factor multiplies BOTH terms as a common prefactor.
    """
    prefactor = (1.0 + nu_val) * p_i * a_r**2 / (E_val * (b_r**2 - a_r**2))
    return prefactor * ((1.0 - 2.0 * nu_val) * r + b_r**2 / r)

def lame_sigma_r(r):
    """Radial stress."""
    return p_i * a_r**2 / (b_r**2 - a_r**2) * (1.0 - b_r**2 / r**2)

def lame_sigma_t(r):
    """Hoop (circumferential) stress."""
    return p_i * a_r**2 / (b_r**2 - a_r**2) * (1.0 + b_r**2 / r**2)

print(f"\nAnalytical values:")
print(f"  u_r(a={a_r}) = {lame_ur(a_r):.6e}")
print(f"  u_r(b={b_r}) = {lame_ur(b_r):.6e}")
print(f"  sigma_r(a)   = {lame_sigma_r(a_r):.4f}  (should be -p_i = {-p_i})")
print(f"  sigma_t(a)   = {lame_sigma_t(a_r):.4f}")
print(f"  max strain   ~ {lame_ur(a_r)/a_r:.4f} ({lame_ur(a_r)/a_r*100:.2f}%)")

# ============================================================
# 3. Gmsh mesh — quarter thick-walled cylinder
# ============================================================
print("\n--- Generating Gmsh mesh ---")
gmsh.initialize()
gmsh.option.setNumber("General.Terminal", 0)
gmsh.model.add("quarter_cylinder")

# Physical group tags
TAG_INNER  = 1   # inner cylindrical surface (r = a)
TAG_OUTER  = 2   # outer cylindrical surface (r = b)
TAG_SYM_Y0 = 3   # symmetry plane y = 0
TAG_SYM_X0 = 4   # symmetry plane x = 0
TAG_Z0     = 5   # end face z = 0
TAG_ZL     = 6   # end face z = L
TAG_VOL    = 7   # volume

occ = gmsh.model.occ

# Build 2D quarter annulus, then extrude to 3D
outer_disk = occ.addDisk(0, 0, 0, b_r, b_r)
inner_disk = occ.addDisk(0, 0, 0, a_r, a_r)
quarter_box = occ.addRectangle(0, 0, 0, b_r + 0.1, b_r + 0.1)

# Intersect outer disk with quarter box (keep first quadrant)
result, _ = occ.intersect([(2, outer_disk)], [(2, quarter_box)],
                          removeObject=True, removeTool=True)
# Cut out inner disk
result, _ = occ.cut(result, [(2, inner_disk)],
                    removeObject=True, removeTool=True)

# Extrude to 3D
occ.extrude(result, 0, 0, L)
occ.synchronize()

# Classify boundary surfaces by geometry
volumes = gmsh.model.getEntities(3)
surfaces = gmsh.model.getEntities(2)

inner_s, outer_s = [], []
sym_y0_s, sym_x0_s = [], []
z0_s, zL_s = [], []

for dim, tag in surfaces:
    bb = gmsh.model.getBoundingBox(dim, tag)
    xmin, ymin, zmin, xmax, ymax, zmax = bb
    com = occ.getCenterOfMass(dim, tag)

    if abs(zmax - zmin) < 1e-6:          # flat in z → end face
        if abs(zmin) < 1e-6:
            z0_s.append(tag)
        else:
            zL_s.append(tag)
    elif abs(ymax - ymin) < 1e-6 and abs(ymin) < 1e-6:  # y = 0 plane
        sym_y0_s.append(tag)
    elif abs(xmax - xmin) < 1e-6 and abs(xmin) < 1e-6:  # x = 0 plane
        sym_x0_s.append(tag)
    else:                                  # curved → inner or outer
        r_com = np.sqrt(com[0]**2 + com[1]**2)
        if r_com < (a_r + b_r) / 2.0:
            inner_s.append(tag)
        else:
            outer_s.append(tag)

print(f"Surface classification:")
print(f"  inner={inner_s}, outer={outer_s}")
print(f"  sym_y0={sym_y0_s}, sym_x0={sym_x0_s}")
print(f"  z0={z0_s}, zL={zL_s}")

# Assign physical groups
gmsh.model.addPhysicalGroup(2, inner_s,   TAG_INNER,  name="inner")
gmsh.model.addPhysicalGroup(2, outer_s,   TAG_OUTER,  name="outer")
gmsh.model.addPhysicalGroup(2, sym_y0_s,  TAG_SYM_Y0, name="sym_y0")
gmsh.model.addPhysicalGroup(2, sym_x0_s,  TAG_SYM_X0, name="sym_x0")
gmsh.model.addPhysicalGroup(2, z0_s,      TAG_Z0,     name="z0")
gmsh.model.addPhysicalGroup(2, zL_s,      TAG_ZL,     name="zL")
gmsh.model.addPhysicalGroup(3, [v[1] for v in volumes], TAG_VOL, name="volume")

# Mesh
gmsh.model.mesh.setSize(gmsh.model.getEntities(0), h)
gmsh.model.mesh.generate(3)

# ============================================================
# 4. Import mesh into dolfinx
# ============================================================
domain, cell_tags, facet_tags = gmshio.model_to_mesh(
    gmsh.model, MPI.COMM_WORLD, 0, gdim=3)
gmsh.finalize()

tdim = domain.topology.dim
fdim = tdim - 1
domain.topology.create_connectivity(fdim, tdim)

n_cells = domain.topology.index_map(tdim).size_global
n_verts = domain.topology.index_map(0).size_global
print(f"\nMesh imported: {n_cells} tetrahedra, {n_verts} vertices")

# ============================================================
# 5. Function space and boundary conditions
# ============================================================
V = fem.functionspace(domain, ("Lagrange", 1, (3,)))
n_dofs = V.dofmap.index_map.size_global * 3
print(f"DOFs: {n_dofs}")

bcs = []

# Symmetry: u_y = 0 on y = 0 plane
facets = facet_tags.find(TAG_SYM_Y0)
dofs = fem.locate_dofs_topological(V.sub(1), fdim, facets)
bcs.append(fem.dirichletbc(default_scalar_type(0.0), dofs, V.sub(1)))

# Symmetry: u_x = 0 on x = 0 plane
facets = facet_tags.find(TAG_SYM_X0)
dofs = fem.locate_dofs_topological(V.sub(0), fdim, facets)
bcs.append(fem.dirichletbc(default_scalar_type(0.0), dofs, V.sub(0)))

# Plane strain: u_z = 0 on z = 0
facets = facet_tags.find(TAG_Z0)
dofs = fem.locate_dofs_topological(V.sub(2), fdim, facets)
bcs.append(fem.dirichletbc(default_scalar_type(0.0), dofs, V.sub(2)))

# Plane strain: u_z = 0 on z = L
facets = facet_tags.find(TAG_ZL)
dofs = fem.locate_dofs_topological(V.sub(2), fdim, facets)
bcs.append(fem.dirichletbc(default_scalar_type(0.0), dofs, V.sub(2)))

print(f"Boundary conditions: 4 Dirichlet (sym_y, sym_x, z0, zL)")

# ============================================================
# 6. Neo-Hookean variational formulation
# ============================================================
u_sol = fem.Function(V, name="displacement")
v_test = ufl.TestFunction(V)

# Kinematics
d = 3
I_mat = ufl.Identity(d)
F_def = I_mat + ufl.grad(u_sol)       # deformation gradient
C_tens = F_def.T * F_def              # right Cauchy-Green
J_det = ufl.det(F_def)                # volume ratio
Ic = ufl.tr(C_tens)                   # first invariant

# Material constants
mu = fem.Constant(domain, default_scalar_type(mu_val))
lmbda = fem.Constant(domain, default_scalar_type(lmbda_val))

# Stored energy: psi = (mu/2)(Ic - 3) - mu*ln(J) + (lambda/2)(ln J)^2
psi = (mu / 2.0) * (Ic - 3) - mu * ufl.ln(J_det) \
    + (lmbda / 2.0) * (ufl.ln(J_det))**2

# Pressure load on inner surface (dead load)
# n = outward normal of domain (points inward on inner surface)
# Internal pressure pushes outward => traction t = -p_i * n
# Potential energy contribution: +p_i * dot(n, u) * ds(inner)
p_load = fem.Constant(domain, default_scalar_type(p_i))
n_facet = ufl.FacetNormal(domain)
ds = ufl.Measure("ds", domain=domain, subdomain_data=facet_tags)

# Total potential energy
Pi = psi * ufl.dx + p_load * ufl.dot(n_facet, u_sol) * ds(TAG_INNER)

# First variation (residual)
F_form = ufl.derivative(Pi, u_sol, v_test)

# ============================================================
# 7. Solve nonlinear problem
# ============================================================
print("\n--- Solving ---")
problem = NonlinearProblem(
    F_form, u_sol, bcs=bcs,
    petsc_options_prefix="neo",
    petsc_options={
        "ksp_type": "preonly",
        "pc_type": "lu",
        "pc_factor_mat_solver_type": "mumps",
        "snes_rtol": 1e-9,
        "snes_atol": 1e-12,
        "snes_max_it": 50,
        "snes_monitor": None,
    })
problem.solve()

n_its = problem.solver.getIterationNumber()
reason = problem.solver.getConvergedReason()
converged = reason > 0
print(f"Newton: {n_its} iterations, converged = {converged} (reason {reason})")

# ============================================================
# 8. Post-processing — comparison with Lamé
# ============================================================
coords = V.tabulate_dof_coordinates()
u_arr = u_sol.x.array.reshape(-1, 3)

# --- Sample along x-axis (y=0, z=0): u_r = u_x ---
mask_xaxis = (np.abs(coords[:, 1]) < 1e-8) & (np.abs(coords[:, 2]) < 1e-8)
r_x = coords[mask_xaxis, 0]
ur_x = u_arr[mask_xaxis, 0]

idx = np.argsort(r_x)
r_x, ur_x = r_x[idx], ur_x[idx]
ur_exact_x = lame_ur(r_x)

abs_err_x = np.abs(ur_x - ur_exact_x)
rel_err_x = abs_err_x / np.abs(ur_exact_x)
l2_err = np.sqrt(np.sum((ur_x - ur_exact_x)**2) / np.sum(ur_exact_x**2))

print("\n" + "=" * 70)
print("FEM vs ANALYTICAL along x-axis (y=0, z=0)")
print("=" * 70)
print(f"{'r':>8s}  {'u_r(FEM)':>14s}  {'u_r(exact)':>14s}  {'rel_err':>12s}")
print("-" * 70)
for i in range(len(r_x)):
    print(f"{r_x[i]:8.4f}  {ur_x[i]:14.6e}  {ur_exact_x[i]:14.6e}  {rel_err_x[i]:12.4e}")
print("-" * 70)
print(f"L2 relative error:  {l2_err:.6e}")
print(f"Max relative error: {rel_err_x.max():.6e}")
print(f"Max absolute error: {abs_err_x.max():.6e}")

# --- Sample along y-axis (x=0, z=0): u_r = u_y ---
mask_yaxis = (np.abs(coords[:, 0]) < 1e-8) & (np.abs(coords[:, 2]) < 1e-8)
r_y = coords[mask_yaxis, 1]
ur_y = u_arr[mask_yaxis, 1]

idx_y = np.argsort(r_y)
r_y, ur_y = r_y[idx_y], ur_y[idx_y]
ur_exact_y = lame_ur(r_y)

l2_err_y = np.sqrt(np.sum((ur_y - ur_exact_y)**2) / np.sum(ur_exact_y**2))
print(f"\nVerification along y-axis: L2 relative error = {l2_err_y:.6e}")

# --- Sample along 45° line (z=0 face) ---
mask_z0 = np.abs(coords[:, 2]) < 1e-8
c_z0 = coords[mask_z0]
u_z0 = u_arr[mask_z0]
r_z0 = np.sqrt(c_z0[:, 0]**2 + c_z0[:, 1]**2)
theta_z0 = np.arctan2(c_z0[:, 1], c_z0[:, 0])

# Points near theta = 45°
mask_45 = np.abs(theta_z0 - np.pi / 4) < np.deg2rad(3)
r_45 = r_z0[mask_45]
ur_45 = (u_z0[mask_45, 0] * np.cos(theta_z0[mask_45]) +
         u_z0[mask_45, 1] * np.sin(theta_z0[mask_45]))
idx_45 = np.argsort(r_45)
r_45, ur_45 = r_45[idx_45], ur_45[idx_45]
ur_exact_45 = lame_ur(r_45)
l2_err_45 = np.sqrt(np.sum((ur_45 - ur_exact_45)**2) / np.sum(ur_exact_45**2))
print(f"Verification along 45° line: L2 relative error = {l2_err_45:.6e}")

# --- Check u_z should be ~0 everywhere (plane strain) ---
max_uz = np.abs(u_arr[:, 2]).max()
print(f"\nMax |u_z| (should be ~0 for plane strain): {max_uz:.6e}")

# --- Global displacement magnitude ---
u_mag = np.linalg.norm(u_arr, axis=1)
print(f"Max |u| = {u_mag.max():.6e}")

# ============================================================
# 9. Plots
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# (a) Radial displacement comparison
r_fine = np.linspace(a_r, b_r, 300)
ax = axes[0]
ax.plot(r_fine, lame_ur(r_fine), 'k-', lw=2, label='Lamé (analytical)')
ax.plot(r_x, ur_x, 'ro', ms=6, label='FEM — x-axis')
if len(r_y) > 0:
    ax.plot(r_y, ur_y, 'b^', ms=5, label='FEM — y-axis')
if len(r_45) > 0:
    ax.plot(r_45, ur_45, 'gs', ms=4, label='FEM — 45° line')
ax.set_xlabel('r')
ax.set_ylabel('u_r')
ax.set_title('Radial Displacement')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (b) Relative error
ax = axes[1]
ax.semilogy(r_x, rel_err_x, 'ro-', ms=5, label='x-axis')
if len(r_y) > 0:
    rel_y = np.abs(ur_y - ur_exact_y) / np.abs(ur_exact_y)
    ax.semilogy(r_y, rel_y, 'b^-', ms=4, label='y-axis')
ax.set_xlabel('r')
ax.set_ylabel('|u_r^FEM − u_r^exact| / |u_r^exact|')
ax.set_title('Relative Error')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# (c) Analytical stress profiles
ax = axes[2]
ax.plot(r_fine, lame_sigma_r(r_fine), 'b-', lw=2, label='σ_r')
ax.plot(r_fine, lame_sigma_t(r_fine), 'r-', lw=2, label='σ_θ')
ax.axhline(0, color='gray', ls='--', alpha=0.5)
ax.set_xlabel('r')
ax.set_ylabel('Stress')
ax.set_title('Lamé Stress Profiles')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.suptitle(
    f'Thick-Walled Cylinder: Neo-Hookean FEM vs Lamé (plane strain)\n'
    f'a={a_r}, b={b_r}, E={E_val}, ν={nu_val}, p={p_i}  |  '
    f'L2 error = {l2_err:.2e}',
    fontsize=12)
plt.tight_layout()
plt.savefig('lame_comparison.png', dpi=150, bbox_inches='tight')
print(f"\nPlot saved: lame_comparison.png")

# ============================================================
# 10. VTK output
# ============================================================
with io.XDMFFile(domain.comm, "displacement.xdmf", "w") as xdmf:
    xdmf.write_mesh(domain)
    xdmf.write_function(u_sol)
print("XDMF output: displacement.xdmf")

# ============================================================
# 11. Summary
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Solver:     FEniCSx (dolfinx) — compressible Neo-Hookean")
print(f"Mesh:       Gmsh quarter-cylinder, h={h}, {n_cells} tets, {n_dofs} DOFs")
print(f"Newton:     {n_its} iterations, converged={converged}")
print(f"L2 error:   {l2_err:.6e}")
print(f"Max rel err:{rel_err_x.max():.6e}")
print(f"Max |u|:    {u_mag.max():.6e}")
print(f"Conclusion: {'PASS — FEM matches Lamé within discretization error' if l2_err < 0.01 else 'CHECK — error larger than expected'}")
