"""Base class for 4C physics module generators.

All physics-specific generators inherit from BaseGenerator and provide
structured domain knowledge, YAML templates, and physics-aware validation
for LLM-driven simulation setup.
"""

from abc import ABC, abstractmethod
from typing import Any


class BaseGenerator(ABC):
    """Abstract base class for all 4C physics module generators.

    Each generator encapsulates domain knowledge for a specific physics module
    (e.g., scalar transport, solid mechanics, peridynamics) and exposes it in
    a structured form that LLMs can consume to produce correct .4C.yaml input
    files.

    Subclasses must define the class-level attributes and implement the four
    abstract methods.
    """

    # ── Class-level metadata (override in every subclass) ──────────────

    module_key: str          # Registry key, e.g. "particle_pd"
    display_name: str        # Human-readable, e.g. "Peridynamics (Bond-Based)"
    problem_type: str        # 4C PROBLEM TYPE value, e.g. "Particle"

    # ── Abstract interface ─────────────────────────────────────────────

    @abstractmethod
    def get_knowledge(self) -> dict[str, Any]:
        """Return structured domain knowledge for LLM consumption.

        Expected keys (all optional but recommended):
            description          – One-paragraph module overview.
            required_sections    – List of mandatory .4C.yaml section names.
            materials            – Dict mapping material name to parameter dict.
            solver               – Solver recommendations (type, tolerances, ...).
            time_integration     – Time-stepping advice.
            pitfalls             – List of common-mistake strings.
            unit_systems         – Dict describing consistent unit sets.
            typical_experiments  – List of dicts with name + description.
        """

    @abstractmethod
    def get_template(self, variant: str = "default") -> str:
        """Return a minimal working YAML template.

        Parameters
        ----------
        variant : str
            Template variant name (see :meth:`list_variants`).

        Returns
        -------
        str
            A complete, runnable .4C.yaml input-file string.

        Raises
        ------
        ValueError
            If *variant* is not recognised by this generator.
        """

    @abstractmethod
    def list_variants(self) -> list[dict[str, str]]:
        """List available template variants.

        Returns
        -------
        list[dict]
            Each dict has at least ``name`` and ``description`` keys.
        """

    @abstractmethod
    def validate_parameters(self, params: dict[str, Any]) -> list[str]:
        """Physics-aware validation of user-supplied parameters.

        Parameters
        ----------
        params : dict
            Arbitrary parameter dict whose contents depend on the generator.

        Returns
        -------
        list[str]
            Human-readable warnings and errors.  An empty list means the
            parameters look physically reasonable.
        """

    # ── Convenience helpers (not usually overridden) ───────────────────

    def get_pitfalls(self) -> list[str]:
        """Return the list of common pitfalls for this physics module."""
        return self.get_knowledge().get("pitfalls", [])

    def get_material_catalog(self) -> dict[str, Any]:
        """Return the materials sub-dictionary from the knowledge base."""
        return self.get_knowledge().get("materials", {})

    def format_knowledge_markdown(self) -> str:
        """Format the full knowledge base as Markdown for LLM consumption.

        Produces a self-contained document covering every section returned
        by :meth:`get_knowledge` so that an LLM can use it as context when
        generating or reviewing 4C input files.
        """
        k = self.get_knowledge()
        lines: list[str] = []

        # ── Title & description ────────────────────────────────────────
        lines.append(f"# {self.display_name}")
        lines.append("")
        lines.append(f"**Module key:** `{self.module_key}`  ")
        lines.append(f"**Problem type:** `{self.problem_type}`")
        lines.append("")

        description = k.get("description", "")
        if description:
            lines.append("## Description")
            lines.append("")
            lines.append(description)
            lines.append("")

        # ── Required sections ──────────────────────────────────────────
        required_sections = k.get("required_sections", [])
        if required_sections:
            lines.append("## Required Input-File Sections")
            lines.append("")
            for section in required_sections:
                lines.append(f"- `{section}`")
            lines.append("")

        # ── Materials ──────────────────────────────────────────────────
        materials = k.get("materials", {})
        if materials:
            lines.append("## Materials")
            lines.append("")
            for mat_name, mat_info in materials.items():
                lines.append(f"### {mat_name}")
                lines.append("")
                if isinstance(mat_info, dict):
                    mat_desc = mat_info.get("description", "")
                    if mat_desc:
                        lines.append(mat_desc)
                        lines.append("")
                    params = mat_info.get("parameters", {})
                    if params:
                        lines.append("| Parameter | Description | Typical Range |")
                        lines.append("|-----------|-------------|---------------|")
                        for pname, pdetail in params.items():
                            if isinstance(pdetail, dict):
                                pdesc = pdetail.get("description", "")
                                prange = pdetail.get("range", "")
                            else:
                                pdesc = str(pdetail)
                                prange = ""
                            lines.append(f"| `{pname}` | {pdesc} | {prange} |")
                        lines.append("")
                elif isinstance(mat_info, str):
                    lines.append(mat_info)
                    lines.append("")

        # ── Solver recommendations ─────────────────────────────────────
        solver = k.get("solver", {})
        if solver:
            lines.append("## Solver Recommendations")
            lines.append("")
            if isinstance(solver, dict):
                for key, value in solver.items():
                    if isinstance(value, dict):
                        lines.append(f"### {key}")
                        lines.append("")
                        for sk, sv in value.items():
                            lines.append(f"- **{sk}:** {sv}")
                        lines.append("")
                    elif isinstance(value, list):
                        lines.append(f"**{key}:**")
                        lines.append("")
                        for item in value:
                            lines.append(f"- {item}")
                        lines.append("")
                    else:
                        lines.append(f"- **{key}:** {value}")
                if not any(isinstance(v, (dict, list)) for v in solver.values()):
                    lines.append("")
            elif isinstance(solver, str):
                lines.append(solver)
                lines.append("")

        # ── Time integration ───────────────────────────────────────────
        time_integration = k.get("time_integration", {})
        if time_integration:
            lines.append("## Time Integration")
            lines.append("")
            if isinstance(time_integration, dict):
                for key, value in time_integration.items():
                    if isinstance(value, dict):
                        lines.append(f"### {key}")
                        lines.append("")
                        for tk, tv in value.items():
                            lines.append(f"- **{tk}:** {tv}")
                        lines.append("")
                    elif isinstance(value, list):
                        lines.append(f"**{key}:**")
                        lines.append("")
                        for item in value:
                            lines.append(f"- {item}")
                        lines.append("")
                    else:
                        lines.append(f"- **{key}:** {value}")
                if not any(isinstance(v, (dict, list)) for v in time_integration.values()):
                    lines.append("")
            elif isinstance(time_integration, str):
                lines.append(time_integration)
                lines.append("")

        # ── Pitfalls ───────────────────────────────────────────────────
        pitfalls = k.get("pitfalls", [])
        if pitfalls:
            lines.append("## Common Pitfalls")
            lines.append("")
            for pitfall in pitfalls:
                lines.append(f"- {pitfall}")
            lines.append("")

        # ── Unit systems ───────────────────────────────────────────────
        unit_systems = k.get("unit_systems", {})
        if unit_systems:
            lines.append("## Consistent Unit Systems")
            lines.append("")
            if isinstance(unit_systems, dict):
                lines.append("| Quantity | " + " | ".join(unit_systems.keys()) + " |")
                lines.append("|----------|" + "|".join("---" for _ in unit_systems) + "|")
                # Collect all quantities across unit systems
                all_quantities: set[str] = set()
                for system in unit_systems.values():
                    if isinstance(system, dict):
                        all_quantities.update(system.keys())
                for qty in sorted(all_quantities):
                    row = f"| {qty} |"
                    for system in unit_systems.values():
                        if isinstance(system, dict):
                            row += f" {system.get(qty, '-')} |"
                        else:
                            row += " - |"
                    lines.append(row)
                lines.append("")
            elif isinstance(unit_systems, list):
                for system in unit_systems:
                    lines.append(f"- {system}")
                lines.append("")

        # ── Typical experiments ────────────────────────────────────────
        typical_experiments = k.get("typical_experiments", [])
        if typical_experiments:
            lines.append("## Typical Experiments / Benchmarks")
            lines.append("")
            for exp in typical_experiments:
                if isinstance(exp, dict):
                    name = exp.get("name", "Unnamed")
                    desc = exp.get("description", "")
                    lines.append(f"### {name}")
                    lines.append("")
                    if desc:
                        lines.append(desc)
                        lines.append("")
                    # Include any additional keys
                    for ek, ev in exp.items():
                        if ek in ("name", "description"):
                            continue
                        if isinstance(ev, list):
                            lines.append(f"**{ek}:**")
                            lines.append("")
                            for item in ev:
                                lines.append(f"- {item}")
                            lines.append("")
                        else:
                            lines.append(f"- **{ek}:** {ev}")
                    # Ensure blank line after each experiment
                    if lines[-1] != "":
                        lines.append("")
                elif isinstance(exp, str):
                    lines.append(f"- {exp}")
            if typical_experiments and isinstance(typical_experiments[-1], str):
                lines.append("")

        # ── Template variants ──────────────────────────────────────────
        variants = self.list_variants()
        if variants:
            lines.append("## Available Template Variants")
            lines.append("")
            for v in variants:
                vname = v.get("name", "unknown")
                vdesc = v.get("description", "")
                lines.append(f"- **`{vname}`** -- {vdesc}")
            lines.append("")

        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<{type(self).__name__} module_key={self.module_key!r}>"
