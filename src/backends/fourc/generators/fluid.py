"""Fluid dynamics (incompressible Navier-Stokes) generator for 4C.

Covers SUPG/PSPG-stabilised incompressible Navier-Stokes on fixed (Euler)
or moving (ALE) grids.  Provides two template variants: a 2-D channel flow
and a 2-D lid-driven cavity.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class FluidGenerator(BaseGenerator):
    """Generator for incompressible Navier-Stokes fluid problems in 4C."""

    module_key = "fluid"
    display_name = "Fluid Dynamics (Incompressible Navier-Stokes)"
    problem_type = "Fluid"

    # ── Knowledge ─────────────────────────────────────────────────────

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "Incompressible Navier-Stokes solver using residual-based "
                "SUPG/PSPG stabilisation.  Supports fixed Eulerian grids "
                "(NA: Euler) and arbitrary-Lagrangian-Eulerian moving grids "
                "(NA: ALE, required for FSI coupling).  Velocity and pressure "
                "are solved in a monolithic system; the pressure DOF is the "
                "last DOF per node (NUMDOF includes pressure)."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "PROBLEM SIZE",
                "FLUID DYNAMIC",
                "SOLVER 1",
                "MATERIALS",
                "FLUID GEOMETRY",
            ],
            "optional_sections": [
                "FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION",
                "FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES",
                "IO/RUNTIME VTK OUTPUT",
                "IO/RUNTIME VTK OUTPUT/FLUID",
            ],
            "materials": {
                "MAT_fluid": {
                    "description": (
                        "Newtonian fluid material.  4C solves incompressible "
                        "Navier-Stokes; density and dynamic viscosity fully "
                        "characterise the fluid."
                    ),
                    "parameters": {
                        "DYNVISCOSITY": {
                            "description": "Dynamic viscosity mu [Pa s]",
                            "range": "> 0 (water ~1e-3, air ~1.8e-5)",
                        },
                        "DENSITY": {
                            "description": "Fluid density rho [kg/m^3]",
                            "range": "> 0 (water ~1000, air ~1.2)",
                        },
                    },
                },
            },
            "solver": {
                "small_2d": {
                    "type": "UMFPACK (direct)",
                    "when": "Small 2-D problems (< ~50 k DOFs)",
                },
                "large_or_3d": {
                    "type": "Belos (iterative) with block preconditioner",
                    "when": "Larger problems or 3-D",
                    "notes": (
                        "Use AZTOL ~1e-8, AZSUB 300. "
                        "Block preconditioner via MueLu XML recommended."
                    ),
                },
            },
            "time_integration": {
                "schemes": [
                    "Np_Gen_Alpha (recommended -- second-order, A-stable, controllable dissipation)",
                    "BDF2 (second-order, strongly A-stable)",
                    "OneStepTheta (first-order with theta=1, i.e. backward Euler)",
                    "Stationary (pseudo-time stepping to steady state)",
                ],
                "key_parameters": {
                    "TIMEINTEGR": "Time integration scheme name",
                    "TIMESTEP": "Time step size dt",
                    "NUMSTEP": "Number of time steps",
                    "MAXTIME": "Maximum simulation time",
                    "ITEMAX": "Max nonlinear iterations per step (default 10)",
                },
            },
            "element_options": {
                "NA": {
                    "Euler": "Fixed grid (standard CFD)",
                    "ALE": "Moving grid (required for FSI)",
                },
                "NUMDOF": {
                    "2D": "3 (vx, vy, pressure)",
                    "3D": "4 (vx, vy, vz, pressure)",
                },
            },
            "dimensionless_numbers": {
                "Reynolds": "Re = rho * U * L / mu  (inertia vs. viscous forces)",
            },
            "pitfalls": [
                "NUMDOF includes the pressure DOF: 3 in 2-D, 4 in 3-D.",
                "Stabilisation (SUPG/PSPG) is critical -- without it, equal-order "
                "elements produce pressure oscillations.",
                "For fully Dirichlet velocity BCs (e.g. lid-driven cavity), the "
                "pressure is determined only up to a constant.  Pin the pressure "
                "at one node or add a single pressure Dirichlet BC.",
                "ONOFF/VAL arrays must have length equal to NUMDOF (3 in 2-D).",
                "Fluid GEOMETRY uses category FLUID (not SOLID) in ELEMENT_BLOCKS.",
                "Use NA: Euler for pure fluid problems, NA: ALE only when mesh moves.",
            ],
            "typical_experiments": [
                {
                    "name": "channel_flow_2d",
                    "description": (
                        "Poiseuille flow in a 2-D channel with parabolic inlet "
                        "velocity.  Good for verifying pressure drop and velocity "
                        "profile against the analytical solution."
                    ),
                },
                {
                    "name": "lid_driven_cavity",
                    "description": (
                        "Enclosed square cavity with a moving lid.  Classic CFD "
                        "benchmark (Ghia et al., 1982).  Tests recirculation, "
                        "pressure field, and stabilisation quality."
                    ),
                },
            ],
        }

    # ── Variants ──────────────────────────────────────────────────────

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "channel_2d",
                "description": (
                    "2-D channel flow (Poiseuille) with parabolic inlet, "
                    "no-slip walls, and natural outflow.  Uses UMFPACK, "
                    "Np_Gen_Alpha, and exodus mesh."
                ),
            },
            {
                "name": "cavity_2d",
                "description": (
                    "2-D lid-driven cavity with unit-velocity lid, no-slip "
                    "walls, and pressure pin.  Classic CFD benchmark."
                ),
            },
        ]

    # ── Templates ─────────────────────────────────────────────────────

    def get_template(self, variant: str = "channel_2d") -> str:
        templates = {
            "channel_2d": self._template_channel_2d,
            "cavity_2d": self._template_cavity_2d,
        }
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_channel_2d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 2-D Channel Flow (Poiseuille)
            #
            # Domain: [0, 6] x [0, 1]  (L=6, H=1)
            # Inlet:  parabolic velocity profile  u(y) = 4*U_max*y*(1-y)
            # Walls:  no-slip (top / bottom)
            # Outlet: natural (do-nothing) boundary
            # ---------------------------------------------------------------
            TITLE:
              - "2-D channel flow (Poiseuille) -- generated template"
            PROBLEM SIZE:
              DIM: 2
            PROBLEM TYPE:
              PROBLEMTYPE: "Fluid"

            # -- Fluid dynamics settings -----------------------------------
            FLUID DYNAMIC:
              LINEAR_SOLVER: 1
              TIMEINTEGR: "Np_Gen_Alpha"
              PREDICTOR: "explicit_second_order_midpoint"
              NUMSTEP: <number_of_steps>
              TIMESTEP: <timestep>
              MAXTIME: <end_time>
              RESTARTEVERY: <restart_interval>
            FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION:
              CHARELELENGTH_PC: "root_of_volume"

            # -- Solver (direct for small 2-D problems) --------------------
            SOLVER 1:
              SOLVER: "UMFPACK"

            # -- Material --------------------------------------------------
            MATERIALS:
              - MAT: 1
                MAT_fluid:
                  DYNVISCOSITY: <dynamic_viscosity>
                  DENSITY: <fluid_density>

            # -- Mesh (exodus file generated separately) -------------------
            FLUID GEOMETRY:
              FILE: "channel_2d.e"
              ELEMENT_BLOCKS:
                - ID: 1
                  FLUID:
                    QUAD4:
                      MAT: 1
                      NA: Euler

            # -- Boundary conditions ---------------------------------------
            # node_set_id 1 = inlet (left edge, x=0)
            DESIGN LINE DIRICH CONDITIONS:
              - E: 1
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [<inlet_velocity>, 0.0, 0.0]
                FUNCT: [1, null, null]
              # node_set_id 2 = bottom wall (y=0)
              - E: 2
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [null, null, null]
              # node_set_id 3 = top wall (y=1)
              - E: 3
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [null, null, null]

            # Parabolic inlet profile: u(y) = 4*y*(1-y)
            FUNCT1:
              - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<inlet_profile_expression>"

            # -- VTK output ------------------------------------------------
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/FLUID:
              OUTPUT_FLUID: true
              VELOCITY: true
              PRESSURE: true
        """)

    @staticmethod
    def _template_cavity_2d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 2-D Lid-Driven Cavity
            #
            # Domain: [0, 1] x [0, 1]
            # Lid:    u_x = 1 on top edge (y=1)
            # Walls:  no-slip on left, right, bottom
            # Pressure: pinned to 0 at bottom-left corner
            # ---------------------------------------------------------------
            TITLE:
              - "2-D lid-driven cavity -- generated template"
            PROBLEM SIZE:
              DIM: 2
            PROBLEM TYPE:
              PROBLEMTYPE: "Fluid"

            # -- Fluid dynamics settings -----------------------------------
            FLUID DYNAMIC:
              LINEAR_SOLVER: 1
              TIMEINTEGR: "Np_Gen_Alpha"
              PREDICTOR: "explicit_second_order_midpoint"
              NUMSTEP: <number_of_steps>
              TIMESTEP: <timestep>
              MAXTIME: <end_time>
              RESTARTEVERY: <restart_interval>
            FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION:
              CHARELELENGTH_PC: "root_of_volume"

            # -- Solver (direct for small 2-D problems) --------------------
            SOLVER 1:
              SOLVER: "UMFPACK"

            # -- Material --------------------------------------------------
            # Re = rho * U * L / mu
            MATERIALS:
              - MAT: 1
                MAT_fluid:
                  DYNVISCOSITY: <dynamic_viscosity>
                  DENSITY: <fluid_density>

            # -- Mesh (exodus file generated separately) -------------------
            FLUID GEOMETRY:
              FILE: "cavity_2d.e"
              ELEMENT_BLOCKS:
                - ID: 1
                  FLUID:
                    QUAD4:
                      MAT: 1
                      NA: Euler

            # -- Boundary conditions ---------------------------------------
            # node_set_id 1 = bottom wall (y=0, no-slip)
            DESIGN LINE DIRICH CONDITIONS:
              - E: 1
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [null, null, null]
              # node_set_id 2 = right wall (x=1, no-slip)
              - E: 2
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [null, null, null]
              # node_set_id 3 = top lid (y=1, u_x = lid_velocity)
              - E: 3
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [<lid_velocity>, 0.0, 0.0]
                FUNCT: [null, null, null]
              # node_set_id 4 = left wall (x=0, no-slip)
              - E: 4
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [1, 1, 0]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [null, null, null]

            # Pin pressure at bottom-left corner (node_set_id 5)
            DESIGN POINT DIRICH CONDITIONS:
              - E: 5
                ENTITY_TYPE: node_set_id
                NUMDOF: 3
                ONOFF: [0, 0, 1]
                VAL: [0.0, 0.0, 0.0]
                FUNCT: [null, null, null]

            # -- VTK output ------------------------------------------------
            IO/RUNTIME VTK OUTPUT:
              INTERVAL_STEPS: <output_interval_steps>
            IO/RUNTIME VTK OUTPUT/FLUID:
              OUTPUT_FLUID: true
              VELOCITY: true
              PRESSURE: true
        """)

    # ── Validation ────────────────────────────────────────────────────

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        """Validate fluid-specific parameters.

        Checks:
        - viscosity > 0
        - density > 0
        - Reynolds number warning if very high (> 5000)
        - NUMDOF consistency with DIM
        """
        issues: list[str] = []

        viscosity = params.get("viscosity") or params.get("DYNVISCOSITY")
        density = params.get("density") or params.get("DENSITY")

        if viscosity is not None:
            try:
                mu = float(viscosity)
                if mu <= 0:
                    issues.append(
                        f"DYNVISCOSITY must be positive, got {mu}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DYNVISCOSITY must be a number, got {viscosity!r}."
                )
        else:
            issues.append(
                "DYNVISCOSITY not provided -- required for MAT_fluid."
            )

        if density is not None:
            try:
                rho = float(density)
                if rho <= 0:
                    issues.append(
                        f"DENSITY must be positive, got {rho}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DENSITY must be a number, got {density!r}."
                )
        else:
            issues.append(
                "DENSITY not provided -- required for MAT_fluid."
            )

        # Reynolds number check
        velocity = params.get("velocity") or params.get("U")
        length = params.get("length") or params.get("L")
        if (
            viscosity is not None
            and density is not None
            and velocity is not None
            and length is not None
        ):
            try:
                mu = float(viscosity)
                rho = float(density)
                U = float(velocity)
                L = float(length)
                if mu > 0:
                    Re = rho * U * L / mu
                    if Re > 5000:
                        issues.append(
                            f"Reynolds number Re = {Re:.0f} is very high.  "
                            f"Consider using finer mesh or turbulence model.  "
                            f"Laminar solver may not converge."
                        )
            except (TypeError, ValueError):
                pass

        # NUMDOF vs DIM
        dim = params.get("DIM") or params.get("dim")
        numdof = params.get("NUMDOF") or params.get("numdof")
        if dim is not None and numdof is not None:
            try:
                d = int(dim)
                n = int(numdof)
                expected = d + 1  # pressure adds one DOF
                if n != expected:
                    issues.append(
                        f"NUMDOF should be {expected} for {d}-D fluid "
                        f"(velocity DOFs + pressure), got {n}."
                    )
            except (TypeError, ValueError):
                pass

        return issues
