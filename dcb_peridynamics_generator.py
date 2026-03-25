#!/usr/bin/env python3
"""
Generator: DCB crack propagation with 4C bond-based peridynamics.

Physics: Double-Cantilever Beam, 2D plane strain, Mode I fracture
Material: PMMA-like (E=3 GPa, G_Ic=400 J/m^2)
Loading: Initial velocity field v_y = sign(y) * V0 * (1 - x/L)
Support: PDFIXED at far end (x ~ L)
Pre-crack: y=0, x in [0, a0]

LEFM reference: G_I = 3 delta^2 E' h^3 / (16 a^4)
  where E' = E/(1-nu^2), delta = CMOD, a = crack length, h = arm height

Units: mm, Mg (tonne), s  =>  stress in MPa, force in N
"""
import math, sys

# ======== Geometry [mm] ========
L   = 80.0       # specimen length
H   = 10.0       # total height (2h)
h   = H / 2.0    # arm height
a0  = 30.0       # initial crack length
B   = 1.0        # out-of-plane thickness (2D unit)

# ======== Material (PMMA) ========
E      = 3000.0      # Young's modulus [MPa]
rho    = 1.18e-9     # density [Mg/mm^3] = 1180 kg/m^3
G_Ic   = 0.4         # fracture energy [N/mm] = 400 J/m^2
nu_pd  = 0.25        # Poisson ratio (BB-PD plane strain, fixed)
E_prim = E / (1.0 - nu_pd**2)  # plane strain modulus = 3200 MPa

# ======== PD discretization ========
dx    = 0.5                    # particle spacing [mm]
m_r   = 3                      # horizon ratio
delta = m_r * dx + 1.0e-4      # horizon = 1.5001 mm (avoid boundary coincidence)

# Critical stretch: s_c = sqrt(5 G_Ic / (9 K delta))
# K = E / [3(1-2nu)] = 2000 MPa for nu=0.25
K_b = E / (3.0 * (1.0 - 2.0 * nu_pd))
s_c = math.sqrt(5.0 * G_Ic / (9.0 * K_b * delta))

# ======== Time integration ========
c_bar = math.sqrt(E / rho)                      # bar wave speed ~ 1.594e6 mm/s
c_s   = math.sqrt(E / (2.0 * (1.0 + nu_pd) * rho))  # shear wave speed
c_R   = 0.92 * c_s                               # Rayleigh wave speed
dt    = 0.4 * dx / c_bar                          # CFL with safety 0.4

# ======== Loading ========
V0    = 1000.0     # initial velocity at x=0 [mm/s]
t_max = 3.0e-3     # simulation time [s]
nstep = int(math.ceil(t_max / dt))
n_out = max(1, nstep // 100)  # ~100 output frames

# ======== LEFM predictions ========
delta_c = math.sqrt(16.0 * a0**4 * G_Ic / (3.0 * E_prim * h**3))
P_c     = math.sqrt(G_Ic * E_prim * B**2 * h**3 / (12.0 * a0**2))

# ======== Domain ========
mg  = 2.0 * delta
bbx = (-mg, L + mg)
bby = (-h - mg, h + mg)

# ======== Generate particles ========
xs = [round(dx/2.0 + i * dx, 6) for i in range(int(round(L / dx)))]
ys = [round(-h + dx/2.0 + j * dx, 6) for j in range(int(round(H / dx)))]
nx, ny = len(xs), len(ys)

# PDFIXED: last 3 columns
n_fix_cols = 3
x_fix = xs[-n_fix_cols]  # threshold x

particles = []
n_fixed = 0
for yi in ys:
    for xi in xs:
        if xi >= x_fix - 1e-10:
            particles.append(
                f'  - "TYPE pdphase POS {xi} {yi} 0.0 PDBODYID 0 PDFIXED 1"'
            )
            n_fixed += 1
        else:
            particles.append(
                f'  - "TYPE pdphase POS {xi} {yi} 0.0 PDBODYID 0"'
            )

n_total = len(particles)

# ======== Summary (stderr) ========
info = f"""# ===== DCB Peridynamics Setup Summary =====
# Geometry: L={L} H={H} a0={a0} h={h} B={B} mm
# Material: E={E} MPa, rho={rho:.2e} Mg/mm^3, G_Ic={G_Ic} N/mm
# PD: nu={nu_pd} (fixed), E'={E_prim:.0f} MPa, K={K_b:.0f} MPa
# Discretization: dx={dx}, delta={delta:.4f}, m={m_r}
# Particles: {n_total} (grid {nx}x{ny}), PDFIXED: {n_fixed}
# Critical stretch: s_c = {s_c:.6f}
# Wave speeds: c_bar={c_bar:.0f}, c_s={c_s:.0f}, c_R={c_R:.0f} mm/s
# Time step: dt={dt:.4e} s, nstep={nstep}, t_max={t_max:.2e} s
# Loading: V0={V0} mm/s, V0/c_bar={V0/c_bar:.2e}
# LEFM: delta_c={delta_c:.3f} mm, P_c={P_c:.3f} N
# Output: every {n_out} steps (~{nstep//n_out} frames)
# ============================================"""
print(info, file=sys.stderr)

# ======== Write YAML (stdout) ========
yaml_out = f"""# DCB crack propagation - 4C peridynamics
# L={L} H={H} a0={a0} E={E} G_Ic={G_Ic} s_c={s_c:.6f}

PROBLEM TYPE:
  PROBLEMTYPE: "Particle"

IO:
  STDOUTEVERY: {n_out}
  VERBOSITY: "Standard"

IO/RUNTIME VTK OUTPUT:
  INTERVAL_STEPS: {n_out}
IO/RUNTIME VTK OUTPUT/PARTICLES:
  PARTICLE_OUTPUT: true
  DISPLACEMENT: true
  VELOCITY: true
  OWNER: true

BINNING STRATEGY:
  BIN_SIZE_LOWER_BOUND: {delta + 1.0:.1f}
  DOMAINBOUNDINGBOX: "{bbx[0]:.1f} {bby[0]:.1f} -0.01 {bbx[1]:.1f} {bby[1]:.1f} 0.01"

PARTICLE DYNAMIC:
  DYNAMICTYPE: "VelocityVerlet"
  INTERACTION: "SPH"
  RESULTSEVERY: {n_out}
  RESTARTEVERY: {n_out * 10}
  TIMESTEP: {dt:.6e}
  NUMSTEP: {nstep}
  MAXTIME: {t_max:.6e}
  GRAVITY_ACCELERATION: "0.0 0.0 0.0"
  PHASE_TO_DYNLOADBALFAC: "pdphase 1.0"
  PHASE_TO_MATERIAL_ID: "pdphase 1"
  RIGID_BODY_MOTION: false
  PD_BODY_INTERACTION: true

PARTICLE DYNAMIC/INITIAL AND BOUNDARY CONDITIONS:
  INITIAL_VELOCITY_FIELD: "pdphase 1"
  CONSTRAINT: "Projection2D"

FUNCT1:
  - COMPONENT: 0
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"
  - COMPONENT: 1
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "(y/(fabs(y)+1.0e-30))*{V0:.1f}*(1.0-x/{L:.1f})"
  - COMPONENT: 2
    SYMBOLIC_FUNCTION_OF_SPACE_TIME: "0.0"

PARTICLE DYNAMIC/SPH:
  KERNEL: QuinticSpline
  KERNEL_SPACE_DIM: Kernel2D
  INITIALPARTICLESPACING: {dx}
  BOUNDARYPARTICLEFORMULATION: AdamiBoundaryFormulation
  TRANSPORTVELOCITYFORMULATION: StandardTransportVelocity

PARTICLE DYNAMIC/PD:
  INTERACTION_HORIZON: {delta:.4f}
  PERIDYNAMIC_GRID_SPACING: {dx}
  PD_DIMENSION: Peridynamic_2DPlaneStrain
  NORMALCONTACTLAW: NormalLinearSpring
  NORMAL_STIFF: {0.06 * E:.1f}
  PRE_CRACKS: "0.0 0.0 {a0:.1f} 0.0"

MATERIALS:
  - MAT: 1
    MAT_ParticlePD:
      INITRADIUS: {dx/2.0}
      INITDENSITY: {rho}
      YOUNG: {E}
      CRITICAL_STRETCH: {s_c:.6f}

PARTICLES:
"""

yaml_out += "\n".join(particles) + "\n"
print(yaml_out)
