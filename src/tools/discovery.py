"""
MCP tools for discovering available solvers and their capabilities.
"""

import json
from mcp.server.fastmcp import FastMCP
from core.registry import list_backends, get_backend, recommend_backend, available_backends


def register_discovery_tools(mcp: FastMCP):

    @mcp.tool()
    def list_solvers() -> str:
        """List all registered FEM solver backends and their availability status.

        Returns information about each backend including:
        - Name and display name
        - Installation status (available / not_installed / misconfigured)
        - Supported physics modules
        - Input format (YAML, Python, C++)
        """
        backends = list_backends()
        if not backends:
            return "No backends registered. Check the server configuration."

        lines = ["# Available FEM Solvers\n"]
        for b in backends:
            status_icon = "OK" if b["status"] == "available" else "NOT INSTALLED"
            lines.append(f"## {b['display_name']} [{status_icon}]")
            lines.append(f"- **Name:** `{b['name']}`")
            lines.append(f"- **Status:** {b['message']}")
            lines.append(f"- **Input format:** {b['input_format']}")
            if b["version"]:
                lines.append(f"- **Version:** {b['version']}")
            lines.append(f"- **Physics:** {', '.join(b['physics'])}")
            lines.append("")

        return "\n".join(lines)

    @mcp.tool()
    def get_solver_info(solver_name: str) -> str:
        """Get detailed information about a specific solver backend.

        Args:
            solver_name: Backend name or alias (e.g. 'fenics', '4C', 'dealii')
        """
        backend = get_backend(solver_name)
        if not backend:
            return f"Unknown solver: {solver_name}. Use list_solvers() to see available options."

        status, message = backend.check_availability()
        physics = backend.supported_physics()

        info = {
            "name": backend.name(),
            "display_name": backend.display_name(),
            "status": status.value,
            "message": message,
            "version": backend.get_version(),
            "input_format": backend.input_format().value,
            "physics_modules": [
                {
                    "name": p.name,
                    "description": p.description,
                    "dims": p.spatial_dims,
                    "elements": p.element_types,
                    "variants": p.template_variants,
                }
                for p in physics
            ],
        }
        return json.dumps(info, indent=2)

    @mcp.tool()
    def recommend_solver_for(physics: str) -> str:
        """Recommend the best available solver for a given physics problem.

        Args:
            physics: Physics type (e.g. 'poisson', 'linear_elasticity', 'heat', 'stokes')

        Returns:
            Recommendation with reasoning.
        """
        backend = recommend_backend(physics)
        if not backend:
            avail = available_backends()
            if not avail:
                return "No FEM solvers are currently available on this machine."
            supported = set()
            for b in avail:
                for p in b.supported_physics():
                    supported.add(p.name)
            return (f"No available solver supports '{physics}'.\n"
                    f"Available physics: {', '.join(sorted(supported))}")

        status, msg = backend.check_availability()
        physics_info = next(
            (p for p in backend.supported_physics() if p.name == physics.lower()),
            None
        )

        result = f"**Recommended:** {backend.display_name()} (`{backend.name()}`)\n"
        result += f"**Status:** {msg}\n"
        result += f"**Input format:** {backend.input_format().value}\n"
        if physics_info:
            result += f"**Dimensions:** {physics_info.spatial_dims}\n"
            result += f"**Variants:** {', '.join(physics_info.template_variants)}\n"
        return result
