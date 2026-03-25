"""Electrochemistry generator for 4C.

Covers electrochemical transport problems governed by the Nernst-Planck
equation with electroneutrality constraints.  Solves for ionic
concentrations and electric potential in electrolyte systems.  Used for
battery electrolyte modeling, rotating disk electrodes, and
diffusion-migration problems.
"""

from __future__ import annotations

import textwrap
from typing import Any

from .base import BaseGenerator


class ElectrochemistryGenerator(BaseGenerator):
    """Generator for electrochemistry (Nernst-Planck) problems in 4C."""

    module_key = "electrochemistry"
    display_name = "Electrochemistry (Nernst-Planck / ELCH)"
    problem_type = "Electrochemistry"

    # -- Knowledge ---------------------------------------------------------

    def get_knowledge(self) -> dict[str, Any]:
        return {
            "description": (
                "The electrochemistry module solves the Nernst-Planck "
                "equation for ionic transport in electrolyte systems.  It "
                "couples diffusion, migration (electric field-driven "
                "transport), and optionally convection of multiple ionic "
                "species.  The electric potential is determined by an "
                "electroneutrality condition.  The PROBLEM TYPE is "
                "'Electrochemistry'.  The module uses SCALAR TRANSPORT "
                "DYNAMIC for the transport equations and ELCH CONTROL for "
                "electrochemistry-specific settings (temperature, "
                "electroneutrality method, diffusion-conduction formulation).  "
                "Materials use MAT_ion for individual ionic species wrapped "
                "in MAT_matlist.  Each ion has a diffusivity and valence."
            ),
            "required_sections": [
                "PROBLEM TYPE",
                "SCALAR TRANSPORT DYNAMIC",
                "SCALAR TRANSPORT DYNAMIC/STABILIZATION",
                "SCALAR TRANSPORT DYNAMIC/NONLINEAR",
                "ELCH CONTROL",
                "SOLVER 1",
                "MATERIALS",
            ],
            "optional_sections": [
                "FLUID DYNAMIC",
                "FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES",
                "SCALAR TRANSPORT DYNAMIC/S2I COUPLING",
                "IO/RUNTIME VTK OUTPUT",
            ],
            "materials": {
                "MAT_ion": {
                    "description": (
                        "Single ionic species material.  Defines the "
                        "diffusion coefficient and charge valence of an "
                        "ion in the electrolyte."
                    ),
                    "parameters": {
                        "DIFFUSIVITY": {
                            "description": (
                                "Diffusion coefficient D_i of the ionic "
                                "species [m^2/s]"
                            ),
                            "range": "> 0",
                        },
                        "VALENCE": {
                            "description": (
                                "Charge number z_i of the ionic species "
                                "(positive for cations, negative for anions)"
                            ),
                            "range": "integer, != 0",
                        },
                    },
                },
                "MAT_matlist": {
                    "description": (
                        "Material list that groups multiple MAT_ion species "
                        "into a single material for the scalar transport "
                        "field.  The number of species determines the "
                        "number of transported scalars."
                    ),
                    "parameters": {
                        "LOCAL": {
                            "description": "Local material flag (typically false)",
                            "range": "true/false",
                        },
                        "NUMMAT": {
                            "description": "Number of ionic species in the list",
                            "range": ">= 2",
                        },
                        "MATIDS": {
                            "description": (
                                "List of MAT_ion material IDs for each species"
                            ),
                            "range": "valid MAT IDs",
                        },
                    },
                },
                "MAT_electrode": {
                    "description": (
                        "Electrode material for Butler-Volmer kinetics at "
                        "electrode-electrolyte interfaces (S2I coupling).  "
                        "Defines concentration-dependent diffusion, "
                        "conductivity, and open-circuit potential."
                    ),
                },
            },
            "solver": {
                "direct": {
                    "type": "UMFPACK",
                    "notes": (
                        "Robust direct solver for electrochemistry.  Works "
                        "well for moderate-size problems."
                    ),
                },
            },
            "time_integration": {
                "SOLVERTYPE": (
                    "'nonlinear' is required for electrochemistry due to "
                    "the nonlinear coupling between concentration and "
                    "potential fields."
                ),
                "TIMESTEP": "Time step size for the transport equation.",
                "NUMSTEP": "Total number of time steps.",
                "MAXTIME": "Maximum simulation time.",
                "THETA": (
                    "Time integration parameter for the one-step-theta "
                    "scheme.  theta=0.5 gives Crank-Nicolson (second-order), "
                    "theta=1.0 gives backward Euler (first-order, more stable)."
                ),
            },
            "elch_settings": {
                "TEMPERATURE": (
                    "Thermodynamic temperature in ELCH CONTROL.  In 4C "
                    "units this is often specified as T/F (temperature "
                    "divided by Faraday constant) for non-dimensionalised "
                    "formulations, e.g. 11604.506 for ~1 V."
                ),
                "EQUPOT": (
                    "Electroneutrality method.  Options: "
                    "'ENC' (electroneutrality constraint -- algebraic), "
                    "'divi' (divergence-based closure equation), "
                    "'Laplace' (Laplace equation for potential)."
                ),
                "DIFFCOND_FORMULATION": (
                    "Set true for concentrated solution theory "
                    "(diffusion-conduction formulation).  Set false for "
                    "dilute solution theory (Nernst-Planck)."
                ),
            },
            "pitfalls": [
                (
                    "EQUPOT in ELCH CONTROL determines how the electric "
                    "potential is computed.  Using the wrong method changes "
                    "the physics entirely.  'ENC' enforces strict "
                    "electroneutrality; 'divi' solves an additional PDE."
                ),
                (
                    "MATID in SCALAR TRANSPORT DYNAMIC must reference the "
                    "MAT_matlist material (not individual MAT_ion entries).  "
                    "The matlist wraps all ionic species."
                ),
                (
                    "The number of transported scalars equals NUMMAT in "
                    "MAT_matlist (one per ionic species) plus one for the "
                    "electric potential (depending on EQUPOT method)."
                ),
                (
                    "Initial conditions for concentrations and potential "
                    "should be specified via INITIALFIELD: 'field_by_function' "
                    "with appropriate INITFUNCNO or STARTFUNCNO.  Each "
                    "scalar component needs its own COMPONENT entry in the "
                    "function."
                ),
                (
                    "CALCFLUX_DOMAIN: 'total' enables post-processing of "
                    "species fluxes.  Without it, flux output is unavailable."
                ),
                (
                    "For S2I (scatra-scatra interface) coupling with "
                    "Butler-Volmer kinetics, both SCALAR TRANSPORT DYNAMIC/"
                    "S2I COUPLING and appropriate DESIGN SURF S2I COUPLING "
                    "CONDITIONS must be specified."
                ),
                (
                    "Stabilisation is typically set to 'no_stabilization' "
                    "for electrochemistry (diffusion-dominated).  Only add "
                    "stabilisation if convection is significant."
                ),
            ],
            "typical_experiments": [
                {
                    "name": "diffusion_migration_3d",
                    "description": (
                        "Diffusion-migration of binary electrolyte in a 3-D "
                        "domain.  Two ionic species with different "
                        "diffusivities and valences.  Tests electroneutrality "
                        "coupling and flux computation."
                    ),
                    "template_variant": "nernst_planck_3d",
                },
            ],
        }

    # -- Variants ----------------------------------------------------------

    def list_variants(self) -> list[dict[str, str]]:
        return [
            {
                "name": "nernst_planck_3d",
                "description": (
                    "3-D Nernst-Planck diffusion-migration problem with "
                    "binary electrolyte (cation + anion).  Uses MAT_ion "
                    "materials in MAT_matlist, ENC electroneutrality, "
                    "UMFPACK solver."
                ),
            },
        ]

    # -- Templates ---------------------------------------------------------

    def get_template(self, variant: str = "nernst_planck_3d") -> str:
        templates = {
            "nernst_planck_3d": self._template_nernst_planck_3d,
        }
        if variant == "default":
            variant = "nernst_planck_3d"
        if variant not in templates:
            available = ", ".join(sorted(templates))
            raise ValueError(
                f"Unknown variant {variant!r}. Available: {available}"
            )
        return templates[variant]()

    @staticmethod
    def _template_nernst_planck_3d() -> str:
        return textwrap.dedent("""\
            # FORMAT TEMPLATE — all numerical values are placeholders.
            # ---------------------------------------------------------------
            # 3-D Nernst-Planck Electrochemistry (Binary Electrolyte)
            #
            # Diffusion-migration of two ionic species (cation and anion)
            # in a 3-D domain with electroneutrality constraint.
            #
            # Mesh: exodus file with:
            #   element_block 1 = electrolyte domain (HEX8 or TET4)
            #   node_set 1 = Dirichlet boundary (fixed concentrations)
            #   node_set 2 = opposite boundary
            # ---------------------------------------------------------------
            TITLE:
              - "3-D electrochemistry (Nernst-Planck) -- generated template"
            PROBLEM TYPE:
              PROBLEMTYPE: "Electrochemistry"

            # == Scalar Transport (carries concentration + potential) ===========
            SCALAR TRANSPORT DYNAMIC:
              SOLVERTYPE: "nonlinear"
              MAXTIME: <end_time>
              NUMSTEP: <number_of_steps>
              TIMESTEP: <timestep>
              RESTARTEVERY: <restart_interval>
              MATID: <matlist_material_id>
              INITIALFIELD: "field_by_function"
              INITFUNCNO: <initial_condition_function_id>
              CALCFLUX_DOMAIN: "total"
              LINEAR_SOLVER: 1
            SCALAR TRANSPORT DYNAMIC/STABILIZATION:
              STABTYPE: "no_stabilization"
            SCALAR TRANSPORT DYNAMIC/NONLINEAR:
              ITEMAX: <max_nonlinear_iterations>
              CONVTOL: <nonlinear_convergence_tolerance>
              EXPLPREDICT: <explicit_predictor_flag>

            # == Electrochemistry control ======================================
            ELCH CONTROL:
              TEMPERATURE: <thermodynamic_temperature>
              EQUPOT: "<electroneutrality_method>"

            # == Solver ========================================================
            SOLVER 1:
              SOLVER: "UMFPACK"
              NAME: "elch_solver"

            # == Materials =====================================================
            MATERIALS:
              # Cation
              - MAT: 1
                MAT_ion:
                  DIFFUSIVITY: <cation_diffusivity>
                  VALENCE: <cation_valence>
              # Anion
              - MAT: 2
                MAT_ion:
                  DIFFUSIVITY: <anion_diffusivity>
                  VALENCE: <anion_valence>
              # Material list wrapping all ionic species
              - MAT: <matlist_material_id>
                MAT_matlist:
                  LOCAL: false
                  NUMMAT: <number_of_species>
                  MATIDS: [1, 2]

            # == Initial condition function ====================================
            # One COMPONENT per scalar: species 1, species 2, potential
            FUNCT<initial_condition_function_id>:
              - COMPONENT: 0
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_concentration_1_expression>"
              - COMPONENT: 1
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_concentration_2_expression>"
              - COMPONENT: 2
                SYMBOLIC_FUNCTION_OF_SPACE_TIME: "<initial_potential_expression>"

            # == Boundary Conditions ===========================================
            DESIGN SURF TRANSPORT DIRICH CONDITIONS:
              - E: <dirichlet_face_id>
                NUMDOF: <num_scalar_dofs>
                ONOFF: [<active_scalar_dofs>]
                VAL: [<boundary_concentrations_and_potential>]
                FUNCT: [<time_functions>]

            # == Geometry ======================================================
            TRANSPORT GEOMETRY:
              FILE: "<mesh_file>"
              ELEMENT_BLOCKS:
                - ID: 1
                  TRANSP:
                    HEX8:
                      MAT: <matlist_material_id>
                      TYPE: Std

            RESULT DESCRIPTION:
              - SCATRA:
                  DIS: "scatra"
                  NODE: <result_node_id>
                  QUANTITY: "phi"
                  VALUE: <expected_concentration>
                  TOLERANCE: <result_tolerance>
        """)

    # -- Validation --------------------------------------------------------

    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        issues: list[str] = []

        # Check diffusivity
        diffusivity = params.get("DIFFUSIVITY")
        if diffusivity is not None:
            try:
                d = float(diffusivity)
                if d <= 0:
                    issues.append(
                        f"DIFFUSIVITY must be > 0, got {d}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"DIFFUSIVITY must be a positive number, "
                    f"got {diffusivity!r}."
                )

        # Check valence
        valence = params.get("VALENCE")
        if valence is not None:
            try:
                z = int(valence)
                if z == 0:
                    issues.append(
                        "VALENCE must be non-zero (charge number of the ion)."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"VALENCE must be a non-zero integer, got {valence!r}."
                )

        # Check EQUPOT
        equpot = params.get("EQUPOT")
        if equpot is not None and equpot not in ("ENC", "divi", "Laplace"):
            issues.append(
                f"EQUPOT must be 'ENC', 'divi', or 'Laplace', "
                f"got {equpot!r}."
            )

        # Check TEMPERATURE
        temperature = params.get("TEMPERATURE")
        if temperature is not None:
            try:
                t = float(temperature)
                if t <= 0:
                    issues.append(
                        f"TEMPERATURE must be > 0 (thermodynamic temperature), "
                        f"got {t}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"TEMPERATURE must be a positive number, "
                    f"got {temperature!r}."
                )

        # Check CONVTOL
        convtol = params.get("CONVTOL")
        if convtol is not None:
            try:
                ct = float(convtol)
                if ct <= 0:
                    issues.append(
                        f"CONVTOL must be > 0, got {ct}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"CONVTOL must be a positive number, got {convtol!r}."
                )

        # Check NUMMAT in matlist
        nummat = params.get("NUMMAT")
        if nummat is not None:
            try:
                nm = int(nummat)
                if nm < 2:
                    issues.append(
                        f"NUMMAT in MAT_matlist should be >= 2 (at least "
                        f"two ionic species for electroneutrality), got {nm}."
                    )
            except (TypeError, ValueError):
                issues.append(
                    f"NUMMAT must be an integer >= 2, got {nummat!r}."
                )

        return issues
