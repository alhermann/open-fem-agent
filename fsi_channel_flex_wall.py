"""
FSI: Channel flow with flexible wall -- Kratos CoSimulation style
=================================================================
Architecture mirrors Kratos CoSimulationApplication:
  - FluidSolverWrapper:  2D Stokes (P1/P1 + PSPG mixed formulation)
  - StructuralSolverWrapper: Euler-Bernoulli beam (clamped-clamped)
  - CouplingScheme: Gauss-Seidel
  - ConvergenceAccelerator: Aitken relaxation
  - DataTransfer: interpolation-based interface mapping

Problem:
  Channel [0, 1.0] x [0, 0.41] m
  Flexible wall: bottom boundary, x in [0.2, 0.8] m
  Inlet: parabolic velocity, U_max = 0.3 m/s
  Outlet: zero-stress (natural BC)

  Fluid:  rho=1000 kg/m^3, mu=10 Pa.s  -->  Re = 12.3 (Stokes regime)
  Wall:   E=2e7 Pa, t=0.01 m, EI=1.667 N.m, clamped at both ends

  Analytical reference: Poiseuille dp = 12*mu*U_avg*L/H^2 = 142.9 Pa
  Expected wall deflection: O(10 mm), ~2-5 % of channel height
"""
import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve
import json

# ================================================================
# Parameters
# ================================================================
L_ch, H_ch = 1.0, 0.41
x_fs, x_fe = 0.2, 0.8
L_flex = x_fe - x_fs

rho_f = 1000.0
mu_f  = 10.0
U_max = 0.3

E_s   = 2.0e7
t_s   = 0.01
EI    = E_s * t_s**3 / 12

nx, ny = 50, 20
n_be   = 30
max_it = 30
tol_c  = 1e-8

Re = rho_f * U_max * H_ch / mu_f
print("=" * 62)
print("  FSI: Channel flow with flexible wall")
print("  Kratos CoSimulation style (Gauss-Seidel + Aitken)")
print("=" * 62)
print(f"  Channel:     {L_ch} x {H_ch} m")
print(f"  Flex wall:   x in [{x_fs}, {x_fe}] m  (length {L_flex} m)")
print(f"  Re = {Re:.1f}   mu = {mu_f}   rho = {rho_f}")
print(f"  EI = {EI:.4f} N.m   t = {t_s*1e3:.0f} mm   E = {E_s:.1e} Pa")
print()


# ================================================================
# FluidSolverWrapper -- Stokes P1/P1 + PSPG stabilization
# ================================================================
class FluidSolverWrapper:
    """2D Stokes solver with mixed P1/P1 + PSPG stabilization.

    DOF layout: [u_0..u_{N-1}, v_0..v_{N-1}, p_0..p_{N-1}]
    System:  [A  -G] [vel] = [f]
             [G^T C] [p  ]   [0]
    where A=viscous, G=pressure gradient, C=PSPG stabilization.
    """

    def __init__(self):
        self.N = (nx + 1) * (ny + 1)
        self.coords = np.zeros((self.N, 2))
        self.elems = []
        self._nm = {}

    def build_mesh(self, wall_w=None):
        n = 0
        for j in range(ny + 1):
            for i in range(nx + 1):
                x0 = i * L_ch / nx
                y0 = j * H_ch / ny
                if wall_w is not None and x_fs <= x0 <= x_fe:
                    w = np.interp(x0, wall_w[0], wall_w[1])
                    y0 = y0 - w * (H_ch - y0) / H_ch
                self.coords[n] = (x0, y0)
                self._nm[(i, j)] = n
                n += 1

        self.elems = []
        for j in range(ny):
            for i in range(nx):
                n1 = self._nm[(i, j)]
                n2 = self._nm[(i+1, j)]
                n3 = self._nm[(i+1, j+1)]
                n4 = self._nm[(i, j+1)]
                self.elems.append((n1, n2, n4))
                self.elems.append((n2, n3, n4))

        nm = self._nm
        self.inlet  = [nm[(0, j)]  for j in range(ny + 1)]
        self.outlet = [nm[(nx, j)] for j in range(ny + 1)]
        self.bottom = [nm[(i, 0)]  for i in range(nx + 1)]
        self.top    = [nm[(i, ny)] for i in range(nx + 1)]
        self.flex   = [nm[(i, 0)]  for i in range(nx + 1)
                       if x_fs - 1e-10 <= i * L_ch / nx <= x_fe + 1e-10]

    def solve(self):
        N = self.N
        ndof = 3 * N
        K = lil_matrix((ndof, ndof))

        for tri in self.elems:
            ids = list(tri)
            x = self.coords[ids, 0]
            y = self.coords[ids, 1]

            a2 = (x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0])
            if abs(a2) < 1e-16:
                continue
            area = abs(a2) / 2.0
            b = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]]) / a2
            c = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]]) / a2

            # PSPG parameter
            h_e = 2.0 * np.sqrt(area / np.pi)
            tau = h_e**2 / (4.0 * mu_f)

            for a in range(3):
                ia = ids[a]
                for bb in range(3):
                    ib = ids[bb]
                    gd = b[a]*b[bb] + c[a]*c[bb]

                    # Viscous stiffness: A (u-u and v-v blocks)
                    kv = mu_f * gd * area
                    K[ia, ib]         += kv           # u-u
                    K[N+ia, N+ib]     += kv           # v-v

                    # Pressure gradient: -G (u-p and v-p blocks)
                    K[ia,   2*N+ib]   -= b[a] * area / 3.0    # u-p
                    K[N+ia, 2*N+ib]   -= c[a] * area / 3.0    # v-p

                    # Divergence: G^T (p-u and p-v blocks)
                    K[2*N+ia, ib]     += area / 3.0 * b[bb]   # p-u
                    K[2*N+ia, N+ib]   += area / 3.0 * c[bb]   # p-v

                    # PSPG stabilization: +C (p-p block)
                    K[2*N+ia, 2*N+ib] += tau * gd * area       # p-p

        # Boundary conditions
        bc = {}
        for n in self.top + self.bottom:
            bc[n] = 0.0;  bc[N + n] = 0.0
        for n in self.inlet:
            yy = self.coords[n, 1]
            bc[n] = U_max * 4.0 * yy * (H_ch - yy) / H_ch**2
            bc[N + n] = 0.0
        # Pin pressure at outlet midpoint
        p_ref = self.outlet[ny // 2]
        bc[2*N + p_ref] = 0.0

        K_csr = K.tocsr()
        F = np.zeros(ndof)
        for dof, val in bc.items():
            F -= np.asarray(K_csr[:, dof].todense()).ravel() * val

        K = K_csr.tolil()
        for dof in bc:
            K[dof, :] = 0;  K[:, dof] = 0;  K[dof, dof] = 1.0
            F[dof] = bc[dof]
        K = K.tocsr()

        sol = spsolve(K, F)
        u = sol[:N]
        v = sol[N:2*N]
        p = sol[2*N:]
        return u, v, p


# ================================================================
# StructuralSolverWrapper -- Euler-Bernoulli beam, clamped-clamped
# ================================================================
class StructuralSolverWrapper:

    def __init__(self, x_nodes):
        self.x = x_nodes
        self.n_el = len(x_nodes) - 1
        self.nn = len(x_nodes)
        self.ndof = 2 * self.nn

    def solve(self, q_vals):
        K = np.zeros((self.ndof, self.ndof))
        F = np.zeros(self.ndof)

        for e in range(self.n_el):
            Le = self.x[e+1] - self.x[e]
            ke = EI / Le**3 * np.array([
                [ 12,     6*Le,   -12,     6*Le   ],
                [ 6*Le,   4*Le**2, -6*Le,  2*Le**2 ],
                [-12,    -6*Le,    12,    -6*Le   ],
                [ 6*Le,   2*Le**2, -6*Le,  4*Le**2 ]])

            q_avg = 0.5 * (q_vals[e] + q_vals[e+1])
            fe = q_avg * np.array([Le/2, Le**2/12, Le/2, -Le**2/12])

            dofs = [2*e, 2*e+1, 2*(e+1), 2*(e+1)+1]
            for i in range(4):
                F[dofs[i]] += fe[i]
                for j in range(4):
                    K[dofs[i], dofs[j]] += ke[i, j]

        fixed = [0, 1, self.ndof - 2, self.ndof - 1]
        free = sorted(set(range(self.ndof)) - set(fixed))

        w_full = np.zeros(self.ndof)
        w_full[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
        return w_full[::2]


# ================================================================
# CoSimulation coupling loop
# ================================================================
beam_x = np.linspace(x_fs, x_fe, n_be + 1)
fluid  = FluidSolverWrapper()
beam   = StructuralSolverWrapper(beam_x)

w_curr = np.zeros(n_be + 1)
r_prev = None
omega  = 0.1       # conservative initial relaxation

print("--- CoSimulation: Gauss-Seidel + Aitken relaxation ---")
print(f"  Fluid mesh:  {(nx+1)*(ny+1)} nodes, {2*nx*ny} triangles, {3*(nx+1)*(ny+1)} DOFs")
print(f"  Beam mesh:   {n_be+1} nodes, {n_be} elements")
print()
print(f"{'It':>3}  {'rel. residual':>13}  {'max w [mm]':>10}  {'omega':>7}  {'avg p [Pa]':>10}")
print("-" * 56)

converged = False
u = v = p = None
history = []

for it in range(max_it):
    # 1. Build / update fluid mesh
    if np.max(np.abs(w_curr)) > 1e-14:
        fluid.build_mesh(wall_w=(beam_x, w_curr))
    else:
        fluid.build_mesh()

    # 2. Solve fluid
    u, v, p = fluid.solve()

    # 3. Data transfer: interface pressure -> beam load
    fx = np.array([fluid.coords[n, 0] for n in fluid.flex])
    fp = np.array([p[n] for n in fluid.flex])
    order = np.argsort(fx)
    q_beam = np.interp(beam_x, fx[order], fp[order])
    avg_p = float(np.mean(q_beam))

    # 4. Solve structure
    w_new = beam.solve(q_beam)

    # 5. Aitken convergence acceleration
    r = w_new - w_curr
    r_norm = np.linalg.norm(r)

    if it >= 2 and r_prev is not None:
        dr = r - r_prev
        drd = np.dot(dr, dr)
        if drd > 1e-20:
            omega = -omega * np.dot(r_prev, dr) / drd
            omega = float(np.clip(omega, 0.05, 0.8))

    w_curr = w_curr + omega * r
    r_prev = r.copy()

    w_max_mm = float(np.max(np.abs(w_curr))) * 1e3
    w_norm = max(np.linalg.norm(w_curr), 1e-12)
    rel_r = r_norm / w_norm

    history.append({"iter": it+1, "rel_residual": rel_r,
                     "max_w_mm": w_max_mm, "omega": omega, "avg_p_Pa": avg_p})
    print(f"{it+1:3d}  {rel_r:13.4e}  {w_max_mm:10.4f}  {omega:7.4f}  {avg_p:10.2f}")

    if rel_r < tol_c:
        converged = True
        break

n_iter = it + 1
status_str = "CONVERGED" if converged else "NOT CONVERGED"
print(f"\n  --> {status_str} in {n_iter} iterations\n")

# ================================================================
# Post-processing
# ================================================================
U_avg = 2.0 / 3.0 * U_max
dp_an  = 12 * mu_f * U_avg * L_ch / H_ch**2
dp_num = float(p[fluid.inlet[ny//2]] - p[fluid.outlet[ny//2]])
u_max_computed = float(np.max(u))
v_max_computed = float(np.max(np.abs(v)))
w_max_final = float(np.max(np.abs(w_curr))) * 1e3

print("--- Results ---")
print(f"  Max wall deflection:  {w_max_final:.4f} mm")
print(f"  Max velocity u:       {u_max_computed:.4f} m/s  (inlet U_max={U_max})")
print(f"  Max |v|:              {v_max_computed:.6f} m/s")
print(f"  Pressure drop:        {dp_num:.2f} Pa  (Poiseuille: {dp_an:.2f} Pa)")
print(f"  Relative dp error:    {abs(dp_num - dp_an)/dp_an*100:.2f}%")

# Wall deflection profile
print(f"\n--- Wall Deflection Profile ---")
print(f"  {'x [m]':>8}  {'w [mm]':>10}  {'p [Pa]':>10}")
for i in range(0, len(beam_x), max(1, len(beam_x) // 15)):
    q_local = float(np.interp(beam_x[i], fx[order], fp[order]))
    print(f"  {beam_x[i]:8.3f}  {w_curr[i]*1e3:10.4f}  {q_local:10.2f}")
if len(beam_x) % max(1, len(beam_x)//15) != 1:
    q_local = float(np.interp(beam_x[-1], fx[order], fp[order]))
    print(f"  {beam_x[-1]:8.3f}  {w_curr[-1]*1e3:10.4f}  {q_local:10.2f}")

# ================================================================
# VTK output
# ================================================================
N = fluid.N
ne = len(fluid.elems)
with open("fsi_channel.vtu", "w") as f:
    f.write('<?xml version="1.0"?>\n')
    f.write('<VTKFile type="UnstructuredGrid" version="0.1">\n<UnstructuredGrid>\n')
    f.write(f'<Piece NumberOfPoints="{N}" NumberOfCells="{ne}">\n')
    f.write('<Points>\n<DataArray type="Float64" NumberOfComponents="3" format="ascii">\n')
    for i in range(N):
        f.write(f"{fluid.coords[i,0]:.8f} {fluid.coords[i,1]:.8f} 0.0\n")
    f.write('</DataArray>\n</Points>\n')
    f.write('<Cells>\n<DataArray type="Int32" Name="connectivity" format="ascii">\n')
    for tri in fluid.elems:
        f.write(f"{tri[0]} {tri[1]} {tri[2]}\n")
    f.write('</DataArray>\n<DataArray type="Int32" Name="offsets" format="ascii">\n')
    for i in range(1, ne + 1):
        f.write(f"{3*i}\n")
    f.write('</DataArray>\n<DataArray type="UInt8" Name="types" format="ascii">\n')
    for _ in range(ne):
        f.write("5\n")
    f.write('</DataArray>\n</Cells>\n')
    f.write('<PointData Vectors="velocity" Scalars="pressure">\n')
    f.write('<DataArray type="Float64" Name="velocity" NumberOfComponents="3" format="ascii">\n')
    for i in range(N):
        f.write(f"{u[i]:.8e} {v[i]:.8e} 0.0\n")
    f.write('</DataArray>\n')
    f.write('<DataArray type="Float64" Name="pressure" format="ascii">\n')
    for i in range(N):
        f.write(f"{p[i]:.8e}\n")
    f.write('</DataArray>\n')
    f.write('<DataArray type="Float64" Name="wall_deflection_mm" format="ascii">\n')
    rev = {}
    for (gi, gj), gn in fluid._nm.items():
        rev[gn] = (gi, gj)
    for i in range(N):
        gi, gj = rev[i]
        xi_orig = gi * L_ch / nx
        if gj == 0 and x_fs <= xi_orig <= x_fe:
            w_here = float(np.interp(xi_orig, beam_x, w_curr)) * 1e3
        else:
            w_here = 0.0
        f.write(f"{w_here:.8e}\n")
    f.write('</DataArray>\n')
    f.write('</PointData>\n')
    f.write('</Piece>\n</UnstructuredGrid>\n</VTKFile>\n')
print(f"\n  VTK output: fsi_channel.vtu")

# ================================================================
# Results summary
# ================================================================
summary = {
    "problem": "FSI: channel flow with flexible wall",
    "solver_style": "Kratos CoSimulation (Gauss-Seidel + Aitken)",
    "fluid_solver": "Stokes P1/P1 + PSPG stabilization",
    "structural_solver": "Euler-Bernoulli beam, clamped-clamped",
    "converged": converged,
    "iterations": n_iter,
    "parameters": {
        "Re": Re, "mu_f": mu_f, "rho_f": rho_f, "U_max": U_max,
        "E_s": E_s, "t_wall_m": t_s, "EI": EI,
        "channel_L_m": L_ch, "channel_H_m": H_ch,
        "flex_wall_m": [x_fs, x_fe],
    },
    "results": {
        "max_wall_deflection_mm": w_max_final,
        "max_velocity_u_ms": u_max_computed,
        "max_abs_velocity_v_ms": v_max_computed,
        "pressure_drop_computed_Pa": dp_num,
        "pressure_drop_analytical_Pa": float(dp_an),
        "dp_relative_error_pct": float(abs(dp_num - dp_an) / dp_an * 100),
    },
    "mesh": {
        "fluid_nodes": N, "fluid_elements": ne,
        "beam_nodes": n_be + 1, "beam_elements": n_be,
    },
    "convergence_history": history,
    "deflection_profile": {
        "x_m": beam_x.tolist(),
        "w_mm": (w_curr * 1e3).tolist(),
    },
}
with open("results_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("  Results JSON: results_summary.json")
