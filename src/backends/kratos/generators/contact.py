"""Kratos contact mechanics generators and knowledge."""


def _contact_2d_kratos(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Contact mechanics via Kratos ContactStructuralMechanicsApplication."""
    return f'''\
"""Contact mechanics — Kratos ContactStructuralMechanicsApplication"""
import json
try:
    import KratosMultiphysics as KM
    import KratosMultiphysics.ContactStructuralMechanicsApplication as CSMA
    print("ContactStructuralMechanicsApplication available")
    summary = {{"note": "Kratos CSMA available — ALM/penalty mortar contact"}}
except ImportError:
    print("ContactStructuralMechanicsApplication not installed")
    summary = {{"note": "ContactStructuralMechanicsApplication not installed"}}
with open("results_summary.json", "w") as _f: json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "contact": {
        "description": "Contact mechanics via ContactStructuralMechanicsApplication",
        "application": "ContactStructuralMechanicsApplication",
        "formulations": ["ALM (Augmented Lagrangian Method)", "Penalty method",
                        "Mortar NTN (Node-to-Node)", "Mortar NTS (Node-to-Segment)"],
        "contact_types": ["Frictionless", "Frictional (Coulomb)"],
        "conditions": ["ALMFrictionlessMortarContact", "ALMFrictionalMortarContact",
                      "PenaltyFrictionlessMortarContact", "PenaltyFrictionalMortarContact"],
        "pitfalls": [
            "Contact surfaces defined as SubModelParts with Conditions",
            "Master/slave designation matters for convergence",
            "ALM penalty parameter needs tuning (too small -> penetration, too large -> ill-conditioning)",
        ],
    },
}

GENERATORS = {
    "contact_2d": _contact_2d_kratos,
}
