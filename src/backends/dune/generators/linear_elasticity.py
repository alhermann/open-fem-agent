"""DUNE-fem linear elasticity generators and knowledge."""

from .poisson import _poisson_2d


def _elasticity_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Linear elasticity — DUNE-fem (placeholder)."""
    return _poisson_2d(params)  # Placeholder


KNOWLEDGE = {
    "linear_elasticity": {
        "description": "Linear elasticity with UFL vector spaces",
        "solver": "galerkin scheme with vector Lagrange space",
        "spaces": "lagrange(gridView, dimRange=2, order=k) for 2D vector",
        "pitfalls": [
            "Use dimRange=2 (or 3) for vector-valued spaces",
            "Lame parameters computed from E and nu as usual",
            "Strain: 0.5*(grad(u) + grad(u).T), Stress: lam*tr(eps)*I + 2*mu*eps",
        ],
    },
}

GENERATORS = {
    "linear_elasticity_2d": _elasticity_2d,
}
