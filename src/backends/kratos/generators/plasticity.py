"""Kratos plasticity generators and knowledge.

Covers the ConstitutiveLawsApplication plasticity framework:
- Yield surfaces: VonMises, DruckerPrager, MohrCoulomb, ModifiedMohrCoulomb, Tresca, Rankine
- Plastic potentials: same set (non-associative flow supported)
- Hardening: perfect, linear, exponential, curve-fitting
- Strain formulations: small strain, finite strain
"""


def _plasticity_3d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE — uniaxial compression with Mohr-Coulomb plasticity.

    Single 3D hex element, displacement-controlled. Demonstrates the
    ConstitutiveLawsApplication plasticity framework."""
    E = params.get("E", 50e6)
    nu = params.get("nu", 0.3)
    c = params.get("c", 50e3)
    phi = params.get("phi", 30.0)
    psi = params.get("psi", 0.1)  # >0 to avoid singular denominator
    max_strain = params.get("max_strain", 0.005)
    n_steps = params.get("n_steps", 100)

    import math
    sin_phi = math.sin(math.radians(phi))
    cos_phi = math.cos(math.radians(phi))
    sigma_y = 2 * c * cos_phi / (1 - sin_phi)

    return f'''\
"""Mohr-Coulomb plasticity: uniaxial compression — Kratos ConstitutiveLawsApplication"""
import sys, json
import numpy as np

import KratosMultiphysics as KM
import KratosMultiphysics.StructuralMechanicsApplication as SMA
import KratosMultiphysics.ConstitutiveLawsApplication as CLA

# Parameters
E = {E}
nu = {nu}
c = {c}
phi_deg = {phi}
psi_deg = {psi}
max_strain = {max_strain}
n_steps = {n_steps}

sin_phi = np.sin(np.radians(phi_deg))
cos_phi = np.cos(np.radians(phi_deg))
sigma_y = 2 * c * cos_phi / (1 - sin_phi)
print(f"Analytical yield stress: {{sigma_y/1e3:.1f}} kPa")

# Create model
model = KM.Model()
mp = model.CreateModelPart("Structure")
mp.AddNodalSolutionStepVariable(KM.DISPLACEMENT)
mp.AddNodalSolutionStepVariable(KM.REACTION)
mp.AddNodalSolutionStepVariable(KM.VOLUME_ACCELERATION)
mp.SetBufferSize(2)

L = 1.0
mp.CreateNewNode(1, 0.0, 0.0, 0.0)
mp.CreateNewNode(2, L,   0.0, 0.0)
mp.CreateNewNode(3, L,   L,   0.0)
mp.CreateNewNode(4, 0.0, L,   0.0)
mp.CreateNewNode(5, 0.0, 0.0, L)
mp.CreateNewNode(6, L,   0.0, L)
mp.CreateNewNode(7, L,   L,   L)
mp.CreateNewNode(8, 0.0, L,   L)

# Material — use CLA variables, NOT KM
prop = mp.CreateNewProperties(1)
prop.SetValue(KM.YOUNG_MODULUS, E)
prop.SetValue(KM.POISSON_RATIO, nu)
prop.SetValue(KM.DENSITY, 0.0)
prop.SetValue(CLA.YIELD_STRESS_COMPRESSION, sigma_y)
prop.SetValue(CLA.YIELD_STRESS_TENSION, sigma_y)
prop.SetValue(CLA.FRICTION_ANGLE, phi_deg)
prop.SetValue(CLA.DILATANCY_ANGLE, psi_deg)
prop.SetValue(KM.FRACTURE_ENERGY, 1e10)
prop.SetValue(CLA.HARDENING_CURVE, 3)  # 3 = perfect plasticity

# Constitutive law — use specific class, NOT the factory with parameters
cl = CLA.SmallStrainIsotropicPlasticity3DModifiedMohrCoulombModifiedMohrCoulomb()
prop.SetValue(KM.CONSTITUTIVE_LAW, cl)

mp.CreateNewElement("SmallDisplacementElement3D8N", 1, [1,2,3,4,5,6,7,8], prop)

dt = 1.0 / n_steps
mp.ProcessInfo[KM.DELTA_TIME] = dt
mp.ProcessInfo[KM.DOMAIN_SIZE] = 3

for node in mp.Nodes:
    node.AddDof(KM.DISPLACEMENT_X, KM.REACTION_X)
    node.AddDof(KM.DISPLACEMENT_Y, KM.REACTION_Y)
    node.AddDof(KM.DISPLACEMENT_Z, KM.REACTION_Z)

# BCs: bottom fixed in z, symmetry planes, top prescribed
for nid in [1,2,3,4]:
    mp.Nodes[nid].Fix(KM.DISPLACEMENT_Z)
    mp.Nodes[nid].SetSolutionStepValue(KM.DISPLACEMENT_Z, 0.0)
for nid in [1,4,5,8]:
    mp.Nodes[nid].Fix(KM.DISPLACEMENT_X)
    mp.Nodes[nid].SetSolutionStepValue(KM.DISPLACEMENT_X, 0.0)
for nid in [1,2,5,6]:
    mp.Nodes[nid].Fix(KM.DISPLACEMENT_Y)
    mp.Nodes[nid].SetSolutionStepValue(KM.DISPLACEMENT_Y, 0.0)
for nid in [5,6,7,8]:
    mp.Nodes[nid].Fix(KM.DISPLACEMENT_Z)

linear_solver = KM.SkylineLUFactorizationSolver()
scheme = KM.ResidualBasedIncrementalUpdateStaticScheme()
convergence_criterion = KM.ResidualCriteria(1e-6, 1e-9)
builder_and_solver = KM.ResidualBasedBlockBuilderAndSolver(linear_solver)
strategy = KM.ResidualBasedNewtonRaphsonStrategy(
    mp, scheme, convergence_criterion, builder_and_solver, 30, True, False, True)
strategy.SetEchoLevel(0)
strategy.Check()

results = {{"steps": [], "strain_zz": [], "stress_zz_kPa": []}}
for step in range(1, n_steps + 1):
    mp.CloneTimeStep(step * dt)
    mp.ProcessInfo[KM.STEP] = step
    u_z = -max_strain * L * step / n_steps
    for nid in [5,6,7,8]:
        mp.Nodes[nid].SetSolutionStepValue(KM.DISPLACEMENT_Z, u_z)
    strategy.Solve()
    stress_vec = mp.Elements[1].CalculateOnIntegrationPoints(KM.PK2_STRESS_VECTOR, mp.ProcessInfo)
    s_avg = np.mean([np.array([s[i] for i in range(6)]) for s in stress_vec], axis=0)
    results["steps"].append(step)
    results["strain_zz"].append(u_z / L)
    results["stress_zz_kPa"].append(s_avg[2] / 1e3)

with open("results_summary.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"Peak stress: {{max(abs(s) for s in results['stress_zz_kPa']):.1f}} kPa (analytical: {{sigma_y/1e3:.1f}} kPa)")
'''


GENERATORS = {
    "plasticity_3d": _plasticity_3d_kratos,
}

KNOWLEDGE = {
    "plasticity": {
        "description": "Elasto-plasticity via ConstitutiveLawsApplication: 7 yield surfaces, 5 plastic potentials, 6 hardening curves",
        "application": "ConstitutiveLawsApplication (pip install KratosConstitutiveLawsApplication)",
        "requires": "StructuralMechanicsApplication (for elements)",
        "yield_surfaces": [
            "VonMises — J2 metal plasticity",
            "Tresca — maximum shear stress",
            "DruckerPrager — smooth cone (soil/concrete)",
            "MohrCoulomb — hexagonal pyramid (classical soil plasticity)",
            "ModifiedMohrCoulomb — regularized MC with tension/compression asymmetry",
            "Rankine — maximum tensile stress (brittle)",
            "SimoJu — damage-type yield for quasi-brittle materials",
        ],
        "plastic_potentials": [
            "VonMises, Tresca, DruckerPrager, MohrCoulomb, ModifiedMohrCoulomb",
            "Non-associative flow: yield surface and plastic potential can differ",
            "Example: MohrCoulomb yield + DruckerPrager potential",
        ],
        "hardening_curves": {
            "0": "LinearSoftening",
            "1": "ExponentialSoftening",
            "2": "InitialHardeningExponentialSoftening",
            "3": "PerfectPlasticity (constant threshold)",
            "4": "CurveFittingHardening (polynomial + exponential)",
            "5": "LinearExponentialSoftening",
            "6": "CurveDefinedByPoints (piecewise-linear)",
        },
        "constitutive_law_naming": {
            "pattern": "<StrainSize><HardeningType><Dimension><YieldSurface><PlasticPotential>",
            "example": "SmallStrainIsotropicPlasticity3DModifiedMohrCoulombModifiedMohrCoulomb",
            "strain_sizes": ["SmallStrain", "FiniteStrain"],
            "hardening_types": ["Isotropic", "Kinematic"],
            "note": "Use the specific class directly in Python API, NOT the factory with Parameters",
        },
        "python_api": {
            "variable_locations": {
                "CLA (ConstitutiveLawsApplication)": [
                    "YIELD_STRESS_COMPRESSION", "YIELD_STRESS_TENSION",
                    "FRICTION_ANGLE", "DILATANCY_ANGLE",
                    "HARDENING_CURVE", "COHESION",
                ],
                "KM (KratosMultiphysics)": [
                    "YOUNG_MODULUS", "POISSON_RATIO", "DENSITY",
                    "FRACTURE_ENERGY", "YIELD_STRESS",
                ],
            },
            "instantiation": (
                "cl = CLA.SmallStrainIsotropicPlasticity3DModifiedMohrCoulombModifiedMohrCoulomb(); "
                "prop.SetValue(KM.CONSTITUTIVE_LAW, cl)"
            ),
            "factory_warning": (
                "SmallStrainIsotropicPlasticityFactory() takes NO constructor arguments. "
                "It reads yield_surface/plastic_potential from Properties in JSON workflow only. "
                "For Python API, use the specific pre-combined class directly."
            ),
        },
        "parameter_translation": {
            "classical_MC_to_modified_MC": {
                "description": "Modified MC uses YIELD_STRESS_COMPRESSION/TENSION instead of cohesion+friction angle",
                "uniaxial_sigma3_eq_0": "YIELD_STRESS_COMPRESSION = 2*c*cos(phi)/(1-sin(phi))",
                "triaxial_with_confining": "YIELD_STRESS_COMPRESSION = 2*c*cos(phi)/(1-sin(phi)) + sigma_3*(1+sin(phi))/(1-sin(phi))",
                "note": "The threshold depends on the stress state — set it for the dominant loading condition",
            },
        },
        "benchmarks": {
            "uniaxial_compression": {
                "description": "Single element uniaxial compression — simplest MC test",
                "analytical_yield": "sigma_y = 2*c*cos(phi)/(1-sin(phi))",
                "example_params": "E=50 MPa, nu=0.3, c=50 kPa, phi=30 deg → sigma_y = 173.2 kPa",
                "reference": "de Souza Neto, Peric, Owen: Computational Methods of Plasticity",
            },
            "triaxial_compression": {
                "description": "Triaxial with confining pressure sigma_3",
                "analytical_yield": "sigma_1 = sigma_3*(1+sin(phi))/(1-sin(phi)) + 2*c*cos(phi)/(1-sin(phi))",
                "example_params": "sigma_3=100 kPa, c=50 kPa, phi=30 deg → sigma_1 = 473.2 kPa, q = 373.2 kPa",
                "reference": "DIANA FEA Mohr-Coulomb Model Verification; validated against PLAXIS",
            },
        },
        "pitfalls": [
            "CRITICAL: Dilatancy angle psi=0 causes singular plastic denominator (dF:C:dG = 0) "
            "at the MC compression meridian (Lode angle = -30 deg). Use psi >= 0.1 deg as workaround, "
            "or implement a principal-stress-space return mapping (Sloan et al., 2001).",
            "MC yield surface has 6 corners in the deviatoric plane. Standard backward Euler in "
            "stress-invariant space needs Lode angle smoothing (switch to Drucker-Prager at |theta| >= 29 deg) "
            "or explicit corner return mapping. Without this, Newton diverges at corners.",
            "Modified MC (Kratos) and Classical MC (textbook) use DIFFERENT parameterizations. "
            "Modified MC uses YIELD_STRESS_COMPRESSION/TENSION; Classical MC uses cohesion + friction angle. "
            "The mapping is stress-state dependent — see parameter_translation above.",
            "For perfect plasticity use HARDENING_CURVE=3 with a large FRACTURE_ENERGY (e.g., 1e10). "
            "Using HARDENING_CURVE=0 (linear) still produces softening unless FRACTURE_ENERGY is very large.",
            "In Python API: constitutive law variables are split across modules (CLA vs KM). "
            "FRICTION_ANGLE, DILATANCY_ANGLE, YIELD_STRESS_COMPRESSION are in CLA; "
            "FRACTURE_ENERGY, YOUNG_MODULUS are in KM. Using the wrong module causes AttributeError.",
            "The factory class SmallStrainIsotropicPlasticityFactory() takes NO constructor arguments. "
            "Passing KM.Parameters to its constructor raises TypeError. Use specific pre-combined classes instead.",
            "SHEAR LOCKING: Linear hex8 (3D8N) elements lock in bending-dominated problems. "
            "For plasticity benchmarks with uniform stress (uniaxial, triaxial), hex8 is fine. "
            "For problems with stress gradients, use quadratic elements (3D20N, 3D27N).",
        ],
        "elements": [
            "SmallDisplacementElement3D8N (linear hex, small strain)",
            "SmallDisplacementElement3D4N (linear tet)",
            "SmallDisplacementElement2D3N/4N (2D plane strain/stress)",
            "TotalLagrangianElement3D8N (finite strain — use with FiniteStrain* laws)",
        ],
    },
}
