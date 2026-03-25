"""Kratos DEM (Discrete Element Method) generators and knowledge."""


def _dem_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE: generates a runnable program. All parameter defaults are placeholders.

    Discrete Element Method simulation for granular material."""
    n_particles = params.get("n_particles", 200)
    radius_min = params.get("radius_min", 0.01)
    radius_max = params.get("radius_max", 0.03)
    density = params.get("density", 2650.0)
    gravity = params.get("gravity", -9.81)
    dt = params.get("dt", 1e-5)
    T_end = params.get("T_end", 0.5)
    young_modulus = params.get("E", 1e7)
    restitution = params.get("restitution", 0.5)
    friction = params.get("friction", 0.4)
    domain_x = params.get("domain_x", 0.5)
    domain_y = params.get("domain_y", 1.0)
    return f'''\
"""Discrete Element Method — granular particle simulation — Kratos (standalone)"""
import numpy as np
import json

# Simulation parameters — set for your problem
n_particles = {n_particles}
r_min, r_max = {radius_min}, {radius_max}
density = {density}
gravity = np.array([0.0, {gravity}])
dt = {dt}
T_end = {T_end}
E_contact = {young_modulus}
e_rest = {restitution}
mu_friction = {friction}
domain_x, domain_y = {domain_x}, {domain_y}

np.random.seed(42)

# Generate particles with random positions and radii
radii = np.random.uniform(r_min, r_max, n_particles)
positions = np.zeros((n_particles, 2))
velocities = np.zeros((n_particles, 2))
forces = np.zeros((n_particles, 2))
masses = density * np.pi * radii**2  # 2D: area * density

# Place particles in a grid pattern within domain
cols = int(np.ceil(np.sqrt(n_particles)))
spacing_x = domain_x / (cols + 1)
spacing_y = domain_y / (cols + 1)
for i in range(n_particles):
    row, col = divmod(i, cols)
    positions[i, 0] = (col + 1) * spacing_x + np.random.uniform(-r_min, r_min)
    positions[i, 1] = domain_y * 0.5 + (row + 1) * spacing_y * 0.5

# Hertzian contact model (normal) + Coulomb friction (tangential)
log_e = np.log(e_rest)
damping_ratio = -log_e / np.sqrt(np.pi**2 + log_e**2)

def compute_contact_forces(pos, vel, rad):
    f = np.zeros_like(pos)
    n_contacts = 0
    for i in range(len(pos)):
        # Wall contacts (floor, left, right walls)
        # Floor (y=0)
        overlap = rad[i] - pos[i, 1]
        if overlap > 0:
            kn = E_contact * np.sqrt(rad[i] * overlap)
            fn = kn * overlap
            vn = vel[i, 1]
            cn = 2 * damping_ratio * np.sqrt(kn * masses[i])
            f[i, 1] += fn - cn * vn
            # Tangential friction
            ft = mu_friction * abs(fn) * np.sign(-vel[i, 0])
            f[i, 0] += ft
            n_contacts += 1
        # Left wall (x=0)
        overlap = rad[i] - pos[i, 0]
        if overlap > 0:
            kn = E_contact * np.sqrt(rad[i] * overlap)
            f[i, 0] += kn * overlap
            n_contacts += 1
        # Right wall (x=domain_x)
        overlap = rad[i] - (domain_x - pos[i, 0])
        if overlap > 0:
            kn = E_contact * np.sqrt(rad[i] * overlap)
            f[i, 0] -= kn * overlap
            n_contacts += 1

        # Particle-particle contacts
        for j in range(i + 1, len(pos)):
            dx = pos[j] - pos[i]
            dist = np.linalg.norm(dx)
            overlap = rad[i] + rad[j] - dist
            if overlap > 0 and dist > 1e-14:
                n_vec = dx / dist
                r_eff = rad[i] * rad[j] / (rad[i] + rad[j])
                m_eff = masses[i] * masses[j] / (masses[i] + masses[j])
                kn = E_contact * np.sqrt(r_eff * overlap)
                fn = kn * overlap
                v_rel = vel[j] - vel[i]
                vn = np.dot(v_rel, n_vec)
                cn = 2 * damping_ratio * np.sqrt(kn * m_eff)
                f_normal = (fn - cn * vn) * n_vec
                # Tangential friction
                vt = v_rel - vn * n_vec
                vt_mag = np.linalg.norm(vt)
                if vt_mag > 1e-14:
                    ft_mag = min(mu_friction * abs(fn), kn * vt_mag * dt)
                    f_tangential = ft_mag * vt / vt_mag
                else:
                    f_tangential = np.zeros(2)
                f[i] -= f_normal + f_tangential
                f[j] += f_normal + f_tangential
                n_contacts += 1
    return f, n_contacts

# Time integration — velocity Verlet
n_steps = int(T_end / dt)
output_interval = max(1, n_steps // 20)
max_ke = 0.0
total_contacts = 0

for step in range(n_steps):
    # Compute forces
    contact_f, n_contacts = compute_contact_forces(positions, velocities, radii)
    forces = contact_f + masses[:, None] * gravity[None, :]
    total_contacts += n_contacts

    # Velocity Verlet integration
    accelerations = forces / masses[:, None]
    positions += velocities * dt + 0.5 * accelerations * dt**2
    velocities += accelerations * dt

    # Kinetic energy tracking
    ke = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
    max_ke = max(max_ke, ke)

    if step % output_interval == 0:
        print(f"Step {{step}}/{{n_steps}}, t={{step*dt:.6f}}, KE={{ke:.6e}}, contacts={{n_contacts}}")

# Final output
ke_final = 0.5 * np.sum(masses * np.sum(velocities**2, axis=1))
print(f"DEM complete: {{n_steps}} steps, final KE={{ke_final:.6e}}")

# Write VTU output for visualization
import meshio
points = np.column_stack([positions, np.zeros(n_particles)])
cells = [("vertex", np.arange(n_particles).reshape(-1, 1))]
point_data = {{
    "radius": radii,
    "velocity_x": velocities[:, 0],
    "velocity_y": velocities[:, 1],
    "velocity_magnitude": np.linalg.norm(velocities, axis=1),
}}
meshio.Mesh(points, cells, point_data=point_data).write("result.vtu")

summary = {{
    "n_particles": n_particles,
    "n_steps": n_steps,
    "dt": dt,
    "final_kinetic_energy": float(ke_final),
    "max_kinetic_energy": float(max_ke),
    "avg_contacts_per_step": float(total_contacts / n_steps),
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("DEM simulation complete.")
'''


KNOWLEDGE = {
    "dem": {
        "description": "Discrete Element Method via Kratos DEMApplication",
        "application": "DEMApplication (pip install KratosDEMApplication)",
        "elements": {
            "3D": ["SphericParticle3D (sphere-sphere contact)", "SphericContinuumParticle3D (bonded)"],
            "2D": ["CylinderParticle2D (disk in 2D)", "CylinderContinuumParticle2D"],
            "cluster": ["Cluster3D (rigid body composed of multiple spheres)"],
        },
        "contact_models": {
            "normal": ["Hertz (elastic)", "linear spring-dashpot", "Hertz-Mindlin"],
            "tangential": ["Coulomb friction", "constant friction", "viscous"],
            "rolling": ["constant torque", "viscous rolling resistance"],
            "bonding": ["parallel bond (beam-like)", "Potyondy-Cundall (2004)"],
        },
        "solver_types": ["explicit (velocity Verlet)", "explicit (symplectic Euler)"],
        "pitfalls": [
            "Time step must satisfy CFL: dt < min(2*sqrt(m/k)) across all contacts",
            "Contact detection: use bounding box bins for O(N) neighbor search",
            "High particle count (>10^5): enable MPI domain decomposition",
            "Particle-wall contacts need separate wall geometry definition",
            "Output: use GiD or VTK with sphere glyph for visualization",
        ],
    },
}

GENERATORS = {
    "dem_2d": _dem_2d_kratos,
}
