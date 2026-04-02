"""
Consolidated MCP tools for the Open FEM Agent.

Reduces 48 tools → ~12 tools by combining related functionality.
Fewer tools = faster schema loading = faster agent response.
"""

import json
import os
import time
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from mcp.server.fastmcp.server import Context
from core.registry import get_backend, available_backends

_OUTPUT_DIR = Path(__file__).resolve().parents[2] / "simulation_outputs"
_COUPLING_DIR = Path(__file__).resolve().parents[2] / "benchmarks" / "coupling"
FOURC_ROOT = Path(os.environ.get("FOURC_ROOT", ""))
_jobs: dict = {}


async def _run_with_progress(ctx: Context, coro, message_prefix: str = "Running"):
    """Run a coroutine while sending periodic MCP progress keepalives.

    This prevents the MCP client from timing out on long-running simulations
    (DUNE JIT compilation, 4C FSI, deal.II builds can take minutes).
    """
    import asyncio

    task = asyncio.create_task(coro)
    elapsed = 0
    try:
        while not task.done():
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=5.0)
            except asyncio.TimeoutError:
                elapsed += 5
                try:
                    await ctx.report_progress(
                        elapsed, total=None,
                        message=f"{message_prefix} ({elapsed}s elapsed)"
                    )
                except Exception:
                    pass  # progress reporting is best-effort
    except Exception:
        if not task.done():
            task.cancel()
        raise
    return task.result()


def _fuzzy_match_physics(backend, query: str) -> str:
    """Fuzzy-match a physics query to an actual physics name in a backend.

    Handles synonyms like 'magnetostatics'→'maxwell', 'thermal'→'heat',
    'elasticity'→'linear_elasticity', 'cfd'→'navier_stokes', etc.
    """
    query_lower = query.lower()

    # Direct match first
    for p in backend.supported_physics():
        if p.name == query_lower:
            return p.name

    # Check if query is substring of any physics name or description
    for p in backend.supported_physics():
        if query_lower in p.name.lower() or query_lower in p.description.lower():
            return p.name

    # Check if any physics name is substring of query
    for p in backend.supported_physics():
        if p.name.lower() in query_lower:
            return p.name

    # Common synonyms
    synonyms = {
        "magnetostatics": "maxwell", "electromagnetics": "maxwell", "em": "maxwell",
        "magnetic": "maxwell", "eddy_current": "maxwell", "nedelec": "maxwell",
        "thermal": "heat", "conduction": "heat", "temperature": "heat",
        "elasticity": "linear_elasticity", "structural": "linear_elasticity",
        "solid": "linear_elasticity", "mechanics": "linear_elasticity",
        "cfd": "navier_stokes", "flow": "navier_stokes", "fluid_dynamics": "navier_stokes",
        "ns": "navier_stokes", "incompressible": "navier_stokes",
        "diffusion": "poisson", "laplace": "poisson", "scalar": "poisson",
        "wave": "helmholtz", "acoustics": "helmholtz",
        "nonlinear_elasticity": "hyperelasticity", "large_deformation": "hyperelasticity",
        "neo_hookean": "hyperelasticity", "finite_strain": "hyperelasticity",
        "vibration": "eigenvalue", "modal": "eigenvalue", "frequencies": "eigenvalue",
        "transport": "convection_diffusion", "advection": "convection_diffusion",
        "plate": "biharmonic", "kirchhoff": "biharmonic",
        "peridynamics": "particle_pd", "pd": "particle_pd", "fracture": "particle_pd",
        "sph": "particle_sph", "smoothed_particle": "particle_sph",
        "thermo_structural": "thermal_structural", "thermomechanical": "thermal_structural",
        "poroelasticity": "porous_media", "poro": "porous_media",
        "consolidation": "porous_media", "terzaghi": "porous_media",
        "biot": "porous_media", "geomechanics": "porous_media",
        "plasticity": "plasticity", "elasto_plasticity": "plasticity",
        "elastoplasticity": "plasticity", "yield": "plasticity",
        "mohr_coulomb": "plasticity", "drucker_prager": "plasticity",
        "von_mises": "plasticity", "j2_plasticity": "plasticity",
        "soil_plasticity": "plasticity", "metal_plasticity": "plasticity",
    }

    mapped = synonyms.get(query_lower, query_lower)
    for p in backend.supported_physics():
        if p.name == mapped:
            return p.name

    # Nothing found — return original
    return query_lower


def _list_alternative_solvers(current_solver: str, physics: str) -> str:
    """List other backends that also support this physics (informational).

    This helps the agent know what alternatives exist if the chosen solver
    runs into issues, without being prescriptive about which to use.
    """
    alternatives = []
    for b in available_backends():
        if b.name() == current_solver:
            continue
        for p in b.supported_physics():
            if p.name == physics or physics in p.name or p.name in physics:
                alternatives.append(f"- **{b.display_name()}**: {p.description}")
                break
    if not alternatives:
        return ""
    return "Other solvers that support this physics:\n" + "\n".join(alternatives)


def register_consolidated_tools(mcp: FastMCP):
    """Register all consolidated tools — ~12 tools instead of 48."""

    # ═══════════════════════════════════════════════════════════
    # 1. KNOWLEDGE (replaces 13 separate knowledge tools)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def knowledge(topic: str, solver: str = "", physics: str = "") -> str:
        """Get knowledge about solvers, physics, materials, coupling, or input formats.

        This is the single entry point for ALL domain knowledge.

        Args:
            topic: What you want to know. Options:
                - "physics" — physics-specific knowledge (needs solver + physics)
                - "pitfalls" — all known pitfalls for a solver
                - "materials" — material catalog for a solver
                - "coupling" — cross-solver coupling knowledge
                - "tsi" — thermo-structural interaction patterns
                - "precice" — preCICE comparison
                - "input_guide" — how to write input files for a solver
                - "solver_guidance" — which solver to use for a physics type
            solver: Backend name (e.g. 'fenics', 'fourc', 'dealii', 'ngsolve')
            physics: Physics type (e.g. 'poisson', 'linear_elasticity', 'navier_stokes')
        """
        if topic == "physics" and solver and physics:
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            k = backend.get_knowledge(physics)
            if not k:
                return f"No knowledge for '{physics}' in {solver}"
            result = json.dumps(k, indent=2, default=str)
            # Append real test file references
            from tools.knowledge import _find_reference_test_files
            ref = _find_reference_test_files(solver, physics)
            if ref:
                result += f"\n\n{ref}"
            return result

        elif topic == "pitfalls" and solver:
            # Try deep knowledge
            try:
                from tools.deep_knowledge import _4C_KNOWLEDGE, _FENICS_KNOWLEDGE
                dicts = {"fourc": _4C_KNOWLEDGE, "4c": _4C_KNOWLEDGE,
                         "fenics": _FENICS_KNOWLEDGE, "fenicsx": _FENICS_KNOWLEDGE}
                d = dicts.get(solver.lower(), {})
                pitfalls = {}
                for k, v in d.items():
                    if isinstance(v, dict) and "pitfalls" in v:
                        pitfalls[k] = v["pitfalls"]
                if pitfalls:
                    return json.dumps(pitfalls, indent=2)
            except ImportError:
                pass
            backend = get_backend(solver)
            if backend:
                all_pitfalls = {}
                for p in backend.supported_physics():
                    k = backend.get_knowledge(p.name)
                    if k and "pitfalls" in k:
                        all_pitfalls[p.name] = k["pitfalls"]
                # Also include general input-format pitfalls (e.g., ExodusII
                # block IDs, FUNCT syntax, shared-node NUMDOF conflict)
                general_k = backend.get_knowledge("input_format")
                if isinstance(general_k, dict):
                    gp = general_k.get("general_pitfalls")
                    if gp:
                        all_pitfalls["general_input_format"] = gp
                    et = general_k.get("element_type_per_physics")
                    if et:
                        all_pitfalls["element_types"] = et
                return json.dumps(all_pitfalls, indent=2)
            return f"No pitfalls found for {solver}"

        elif topic == "materials" and solver:
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            materials = {}
            for p in backend.supported_physics():
                k = backend.get_knowledge(p.name)
                if k and "materials" in k:
                    materials[p.name] = k["materials"]
            return json.dumps(materials, indent=2) if materials else f"No material catalog for {solver}"

        elif topic == "coupling":
            from tools.knowledge import register_knowledge_tools
            # Return coupling knowledge directly
            return _get_coupling_knowledge()

        elif topic == "tsi":
            return _get_tsi_knowledge()

        elif topic == "precice":
            return _get_precice_knowledge()

        elif topic == "input_guide" and solver:
            from tools.examples_search import (
                _4C_INPUT_GUIDE, _FENICS_INPUT_GUIDE, _DEALII_INPUT_GUIDE,
                _FEBIO_INPUT_GUIDE, _DUNE_INPUT_GUIDE,
            )
            guides = {"fourc": _4C_INPUT_GUIDE, "4c": _4C_INPUT_GUIDE,
                      "fenics": _FENICS_INPUT_GUIDE, "dealii": _DEALII_INPUT_GUIDE,
                      "febio": _FEBIO_INPUT_GUIDE,
                      "dune": _DUNE_INPUT_GUIDE, "dune-fem": _DUNE_INPUT_GUIDE,
                      "dunefem": _DUNE_INPUT_GUIDE}
            return guides.get(solver.lower(), f"No input guide for {solver}")

        elif topic == "solver_guidance" and physics:
            results = {}
            for b in available_backends():
                for p in b.supported_physics():
                    if p.name == physics:
                        results[b.display_name()] = {
                            "variants": p.template_variants,
                            "elements": p.element_types,
                            "dims": p.spatial_dims,
                        }
            return json.dumps(results, indent=2) if results else f"No solver supports '{physics}'"

        else:
            return (
                "Usage: knowledge(topic, solver, physics)\n"
                "Topics: physics, pitfalls, materials, coupling, tsi, precice, "
                "input_guide, solver_guidance"
            )

    # ═══════════════════════════════════════════════════════════
    # 2. DISCOVER (replaces 6 discovery tools)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def discover(query: str = "list", solver: str = "") -> str:
        """Discover available solvers and their capabilities.

        Args:
            query: What to discover. Options:
                - "list" — list all solvers with status
                - "physics" — list all physics types per solver
                - "capabilities" — full capabilities matrix
                - "recommend" — recommend solver for a physics (set solver= to physics name)
            solver: Filter by solver name, or physics name for "recommend"
        """
        if query == "list":
            lines = []
            for b in available_backends():
                status, msg = b.check_availability()
                lines.append(f"- **{b.display_name()}** ({b.name()}): {status.value} — {b.input_format().value} input")
            return "\n".join(lines) if lines else "No backends available."

        elif query == "physics":
            lines = []
            backends = [get_backend(solver)] if solver else available_backends()
            backends = [b for b in backends if b]
            for b in backends:
                lines.append(f"## {b.display_name()}")
                for p in b.supported_physics():
                    lines.append(f"- **{p.name}**: {p.description} (variants: {', '.join(p.template_variants)})")
                lines.append("")
            return "\n".join(lines)

        elif query == "capabilities":
            lines = ["| Solver | Physics Count | Input | Status |",
                     "|--------|--------------|-------|--------|"]
            for b in available_backends():
                status, _ = b.check_availability()
                lines.append(f"| {b.display_name()} | {len(b.supported_physics())} | {b.input_format().value} | {status.value} |")
            return "\n".join(lines)

        elif query == "recommend":
            physics = solver  # in this case solver param holds the physics name
            results = []
            for b in available_backends():
                for p in b.supported_physics():
                    if physics.lower() in p.name.lower() or physics.lower() in p.description.lower():
                        results.append(f"- **{b.display_name()}**: {p.description}")
                        break
            return "\n".join(results) if results else f"No solver found for '{physics}'"

        return "Usage: discover(query='list'|'physics'|'capabilities'|'recommend', solver='')"

    # ═══════════════════════════════════════════════════════════
    # 3. EXAMPLES (replaces 7 example/search tools)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def examples(keyword: str, solver: str = "fourc", action: str = "search",
                 max_results: int = 3) -> str:
        """Find and retrieve example input files from solver test suites.

        IMPORTANT: Always call this before writing new input files to study
        real, validated configurations.

        Args:
            keyword: Search term (e.g. 'peridynamic', 'fsi', 'poisson', 'heat')
            solver: Backend name (default: 'fourc')
            action: What to do. Options:
                - "search" — find matching test files with content preview
                - "template" — get a generated template for this physics
                - "tutorials" — list available tutorials
            max_results: Maximum results (default 3)
        """
        if action == "search":
            from tools.examples_search import register_example_tools
            # Search real test files
            results = []
            test_dirs = {
                "fourc": FOURC_ROOT / "tests" / "input_files" if FOURC_ROOT else None,
                "4c": FOURC_ROOT / "tests" / "input_files" if FOURC_ROOT else None,
                "dealii": Path("/usr/share/doc/libdeal.ii-doc/examples"),
            }
            solver_key = solver.lower()
            test_dir = test_dirs.get(solver_key)
            ext = "*.4C.yaml" if solver_key in ("fourc", "4c") else "*.cc" if solver_key == "dealii" else "*.py"

            if test_dir and test_dir.is_dir():
                for f in sorted(test_dir.rglob(ext)):
                    if keyword.lower() in f.name.lower():
                        try:
                            content = f.read_text()[:5000]
                            rel = f.relative_to(test_dir)
                            results.append(f"### `{rel}`\n```\n{content}\n```\n")
                        except Exception:
                            pass
                        if len(results) >= max_results:
                            break

            # Also search templates
            backend = get_backend(solver)
            if backend:
                for p in backend.supported_physics():
                    if keyword.lower() in p.name.lower() or keyword.lower() in p.description.lower():
                        for v in p.template_variants[:1]:
                            try:
                                content = backend.generate_input(p.name, v, {})
                                results.append(f"### Template: `{p.name}/{v}`\n```\n{content[:3000]}\n```\n")
                            except Exception:
                                pass
                        break

            if not results:
                return f"No examples found for '{keyword}' in {solver}"
            return f"## {len(results)} example(s) for '{keyword}' from {solver}\n\n" + "\n---\n".join(results)

        elif action == "template":
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            for p in backend.supported_physics():
                if keyword.lower() in p.name.lower():
                    variant = p.template_variants[0] if p.template_variants else "2d"
                    try:
                        content = backend.generate_input(p.name, variant, {})
                        fmt = backend.input_format().value
                        return f"```{fmt}\n{content}\n```"
                    except Exception as e:
                        return f"Error generating template: {e}"
            return f"No template for '{keyword}' in {solver}"

        elif action == "tutorials":
            backend = get_backend(solver)
            if not backend:
                return f"Unknown solver: {solver}"
            lines = [f"## {backend.display_name()} Templates\n"]
            for p in backend.supported_physics():
                lines.append(f"- **{p.name}**: {', '.join(p.template_variants)} — {p.description}")
            return "\n".join(lines)

        return "Usage: examples(keyword, solver, action='search'|'template'|'tutorials')"

    # ═══════════════════════════════════════════════════════════
    # 4. SIMULATE (replaces run_simulation + run_with_generator)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    async def run_with_generator(solver: str, generator_script: str,
                                  job_name: str = "", np: int = 1,
                                  critic_approved: bool = False,
                                  ctx: Context = None) -> str:
        """Run a generator script that creates an input file, then execute the solver.

        Use this for solvers that need a COMPILED binary or separate input files:
        - 4C: generator creates .4C.yaml + mesh, then 4C binary runs on them
        - deal.II: generator creates main.cpp, then cmake + make + ./fem_solve
        - Kratos (with real binary): generator creates ProjectParameters.json +
          .mdpa + MainKratos.py, then Kratos Python runs MainKratos.py

        DO NOT use this for:
        - FEniCS, NGSolve, scikit-fem, DUNE-fem: use run_simulation() instead
        - Kratos manual-assembly scripts (numpy/scipy): use run_simulation()
          since those are standalone Python scripts, not input-file generators

        The generator script runs in the server's Python. It must produce an
        input file matching one of: *.4C.yaml, *.yaml, input.*, solve.py,
        MainKratos.py

        Args:
            solver: Backend name (fourc, dealii, kratos)
            generator_script: Python script that creates the input file
            job_name: Optional job directory name
            np: MPI processes (default 1)
            critic_approved: Set True only after critic agent approved setup
        """
        import subprocess
        import sys

        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        status, msg = backend.check_availability()
        if status.value != "available":
            return f"Solver {solver} not available: {msg}"

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = job_name or f"{solver}_gen_{ts}"
        work_dir = _OUTPUT_DIR / name
        work_dir.mkdir(parents=True, exist_ok=True)

        gen_path = work_dir / "generate_input.py"
        gen_path.write_text(generator_script)

        python = sys.executable
        gen_result = subprocess.run(
            [python, str(gen_path)],
            capture_output=True, text=True,
            cwd=str(work_dir),
        )

        if gen_result.returncode != 0:
            return json.dumps({
                "status": "failed", "phase": "generator",
                "error": gen_result.stderr[-500:],
                "work_dir": str(work_dir),
            }, indent=2)

        input_file = None
        for pattern in ["*.4C.yaml", "*.yaml", "input.*", "solve.py"]:
            matches = list(work_dir.glob(pattern))
            if matches:
                input_file = matches[0]
                break

        if not input_file:
            return json.dumps({
                "status": "failed", "phase": "generator",
                "error": "Generator did not produce an input file",
                "work_dir": str(work_dir),
            }, indent=2)

        input_content = input_file.read_text()
        from core.backend import JobHandle
        run_coro = backend.run(input_content, work_dir, np=np, timeout=None)
        if ctx is not None:
            job = await _run_with_progress(ctx, run_coro, f"Running {solver}")
        else:
            job = await run_coro
        _jobs[job.job_id] = job

        result = {
            "job_id": job.job_id, "solver": solver,
            "status": job.status, "work_dir": str(job.work_dir),
            "elapsed": f"{job.elapsed:.2f}s" if job.elapsed else None,
            "input_file": input_file.name,
            "critic_review": "approved" if critic_approved else "SKIPPED",
        }
        if job.error:
            result["error"] = job.error[:500]
        if job.status == "completed":
            result["output_files"] = [f.name for f in backend.get_result_files(job)]
            stdout_log = work_dir / "stdout.log"
            if stdout_log.exists():
                text = stdout_log.read_text()
                result["stdout_tail"] = text[-2000:] if len(text) > 2000 else text
        return json.dumps(result, indent=2)

    @mcp.tool()
    async def run_simulation(solver: str, input_content: str,
                             job_name: str = "", np: int = 1,
                             critic_approved: bool = False,
                             ctx: Context = None) -> str:
        """Run a simulation directly with input content.

        Use this for Python-based solvers (FEniCS, NGSolve, scikit-fem, DUNE-fem)
        where the input IS a Python script. The tool routes through the correct
        Python environment automatically (e.g., conda env for FEniCS).

        For 4C/deal.II/Kratos where a separate input file must be generated
        first, use run_with_generator() instead.

        Args:
            solver: Backend name (best for: fenics, ngsolve, skfem, dune)
            input_content: The input content (Python script / YAML / C++ / XML)
            job_name: Optional job name
            np: MPI processes
            critic_approved: Set True only after critic agent approved
        """
        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        status, msg = backend.check_availability()
        if status.value != "available":
            return f"Solver {solver} not available: {msg}"

        _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        name = job_name or f"{solver}_{ts}"
        work_dir = _OUTPUT_DIR / name
        work_dir.mkdir(parents=True, exist_ok=True)

        run_coro = backend.run(input_content, work_dir, np=np, timeout=None)
        if ctx is not None:
            job = await _run_with_progress(ctx, run_coro, f"Running {solver}")
        else:
            job = await run_coro
        _jobs[job.job_id] = job

        result = {
            "job_id": job.job_id, "solver": solver,
            "status": job.status, "work_dir": str(job.work_dir),
            "elapsed": f"{job.elapsed:.2f}s" if job.elapsed else None,
            "critic_review": "approved" if critic_approved else "SKIPPED",
        }
        if job.error:
            result["error"] = job.error[:500]
        if job.status == "completed":
            result["output_files"] = [f.name for f in backend.get_result_files(job)]
            # Include stdout so the agent sees script output directly
            stdout_log = work_dir / "stdout.log"
            if stdout_log.exists():
                text = stdout_log.read_text()
                result["stdout_tail"] = text[-2000:] if len(text) > 2000 else text
        return json.dumps(result, indent=2)

    # ═══════════════════════════════════════════════════════════
    # 5. COUPLING (keep as-is — core workflow)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    async def coupled_solve(
        problem: str = "heat_dd", solver_a: str = "fenics",
        solver_b: str = "fourc", nx: int = 32, ny: int = 32,
        max_iter: int = 20, tol: float = 1e-6,
        relaxation: float = 1.0, params: str = "{}",
        critic_approved: bool = False,
    ) -> str:
        """Cross-solver domain decomposition coupling.

        Domain A (Dirichlet at interface) supports: fenics, ngsolve, skfem, dune.
        Domain B (Neumann at interface) supports: fenics, fourc, ngsolve, skfem, dune.
        Any combination of these works for heat_dd and poisson_dd problems.

        Args:
            problem: 'heat_dd', 'poisson_dd', 'one_way', 'tsi_dd',
                     'poisson_dd_study', 'l_bracket_tsi', 'heat_dd_precice'
            solver_a, solver_b: Backend names
            nx, ny: Elements per direction
            max_iter: Max iterations
            tol: Convergence tolerance
            relaxation: Under-relaxation parameter
            params: JSON with additional parameters
            critic_approved: Set True after critic review
        """
        # Import and delegate to the full coupling implementation
        from tools.coupling import register_coupling_tools
        # The coupling tools are complex — delegate to the original implementation
        from tools.coupling import (
            _heat_domain_decomposition, _poisson_domain_decomposition,
            _oneway_thermal_structural, _twoway_tsi_coupling,
            _relaxation_parameter_study, _l_bracket_tsi,
            _heat_dd_precice_comparison,
        )

        param_dict = json.loads(params)
        backend_a = get_backend(solver_a)
        backend_b = get_backend(solver_b)
        if not backend_a or not backend_b:
            return f"Backend not found: {solver_a} or {solver_b}"

        dispatch = {
            "heat_dd": lambda: _heat_domain_decomposition(backend_a, backend_b, nx, ny, max_iter, tol, relaxation, param_dict),
            "poisson_dd": lambda: _poisson_domain_decomposition(backend_a, backend_b, nx, ny, max_iter, tol, relaxation, param_dict),
            "one_way": lambda: _oneway_thermal_structural(backend_a, backend_b, nx, ny, param_dict),
            "tsi_dd": lambda: _twoway_tsi_coupling(backend_a, backend_b, nx, ny, max_iter, tol, relaxation, param_dict),
            "poisson_dd_study": lambda: _relaxation_parameter_study(backend_a, backend_b, nx, ny, max_iter, tol, param_dict),
            "l_bracket_tsi": lambda: _l_bracket_tsi(backend_a, backend_b, nx, ny, param_dict),
            "heat_dd_precice": lambda: _heat_dd_precice_comparison(backend_a, backend_b, nx, ny, max_iter, tol, relaxation, param_dict),
        }

        if problem not in dispatch:
            return f"Unknown problem: {problem}. Available: {list(dispatch.keys())}"

        return await dispatch[problem]()

    # ═══════════════════════════════════════════════════════════
    # 6. VISUALIZE (replaces 4 visualization tools)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    async def visualize(job_id: str = "", work_dir: str = "",
                        action: str = "summary", field: str = "",
                        ctx: Context = None) -> str:
        """Post-process and visualize simulation results.

        Args:
            job_id: Job ID from run_simulation (or leave empty and set work_dir)
            work_dir: Direct path to results directory
            action: 'summary' (field stats), 'plot' (generate PNG), 'list' (show files)
            field: Specific field name to plot (e.g. 'temperature', 'displacement')
        """
        from core.backend import JobHandle

        # Find work directory
        if job_id and job_id in _jobs:
            wd = _jobs[job_id].work_dir
        elif work_dir:
            wd = Path(work_dir)
        else:
            return "Provide job_id or work_dir"

        if not wd.is_dir():
            return f"Directory not found: {wd}"

        # Collect result files — skip .pvtu (parallel wrappers that can hang PyVista)
        vtu_files = [f for f in sorted(wd.rglob("*.vtu")) if not f.name.endswith(".pvtu")]
        vtu_files += sorted(wd.rglob("*.vtk"))
        vtu_files += sorted(wd.rglob("*.vtp"))
        vtu_files += sorted(wd.rglob("*.xdmf"))
        vtu_files += sorted(wd.rglob("*.bp"))  # ADIOS2/VTX output from dolfinx 0.10+

        if action == "list":
            return "\n".join(f"- {f.relative_to(wd)}" for f in vtu_files) or "No VTU/VTP files found"

        elif action == "summary":
            try:
                from core.post_processing import read_mesh
                import numpy as np
                import re

                # Group VTU files by field type (structure, fluid, ale, etc.)
                # 4C multi-physics outputs separate files per field
                field_groups: dict[str, list] = {}
                for vtu in vtu_files:
                    name = vtu.stem
                    # Detect field type from filename patterns like
                    # structure-00-0, fluid-05-0, ale-03-0
                    match = re.match(r'^(.*?)(?:-\d+)?(?:-\d+)?$', name)
                    group_name = match.group(1) if match else name
                    # Also strip trailing -0 (processor rank)
                    group_name = re.sub(r'-\d+$', '', group_name)
                    field_groups.setdefault(group_name, []).append(vtu)

                def _safe_float(v):
                    """Convert to float, replacing NaN/Inf with string markers."""
                    f = float(v)
                    if np.isnan(f):
                        return "NaN"
                    if np.isinf(f):
                        return "Inf" if f > 0 else "-Inf"
                    return f

                results = []
                # Show summary per field group, using the last timestep
                # Limit to 10 groups to avoid extremely long responses
                group_idx = 0
                for group, files in sorted(field_groups.items())[:10]:
                    group_idx += 1
                    if ctx is not None:
                        try:
                            await ctx.report_progress(
                                group_idx, len(field_groups),
                                f"Reading {group} ({len(files)} timesteps)")
                        except Exception:
                            pass
                    # Use the last file in each group (latest timestep)
                    last_vtu = sorted(files)[-1]
                    try:
                        mesh = read_mesh(last_vtu)
                        fields = {}
                        for fname in mesh.point_data:
                            arr = np.asarray(mesh.point_data[fname])
                            n_nan = int(np.isnan(arr).sum())
                            n_inf = int(np.isinf(arr).sum())
                            finite = arr[np.isfinite(arr)]
                            stats = {
                                "shape": list(arr.shape),
                            }
                            if len(finite) > 0:
                                stats["min"] = _safe_float(finite.min())
                                stats["max"] = _safe_float(finite.max())
                                stats["mean"] = _safe_float(finite.mean())
                            if n_nan > 0:
                                stats["WARNING_NaN"] = f"{n_nan} values"
                            if n_inf > 0:
                                stats["WARNING_Inf"] = f"{n_inf} values"
                            fields[fname] = stats
                        results.append({
                            "field_group": group,
                            "timesteps": len(files),
                            "latest_file": last_vtu.name,
                            "points": mesh.n_points,
                            "fields": fields,
                        })
                    except Exception as e:
                        results.append({
                            "field_group": group,
                            "timesteps": len(files),
                            "error": str(e),
                        })
                return json.dumps(results, indent=2)
            except Exception as e:
                return f"Error reading results: {e}"

        elif action == "plot" and field:
            try:
                from core.post_processing import read_mesh, plot_field
                vtu = vtu_files[-1] if vtu_files else None
                if not vtu:
                    return "No VTU files to plot"
                mesh = read_mesh(vtu)
                plot_path = wd / f"plot_{field}.png"
                plot_field(mesh, field, plot_path, title=field, spatial_dim=2)
                return f"Plot saved: {plot_path}"
            except Exception as e:
                return f"Error plotting: {e}"

        elif action == "validate":
            # Automated sanity checks on results
            try:
                from core.post_processing import read_mesh
                import numpy as np
                checks = []
                for vtu in vtu_files[:3]:
                    mesh = read_mesh(vtu)
                    for name in mesh.point_data:
                        arr = np.asarray(mesh.point_data[name])
                        issues = []
                        if np.any(np.isnan(arr)):
                            issues.append(f"CONTAINS NaN ({np.isnan(arr).sum()} values)")
                        if np.any(np.isinf(arr)):
                            issues.append(f"CONTAINS Inf ({np.isinf(arr).sum()} values)")
                        if arr.max() == arr.min() and len(arr) > 1:
                            issues.append(f"CONSTANT FIELD (all values = {arr.max():.6e})")
                        if arr.max() > 1e15:
                            issues.append(f"SUSPICIOUSLY LARGE max = {arr.max():.2e}")
                        status = "PASS" if not issues else "ISSUES FOUND"
                        checks.append(f"- {name} in {vtu.name}: {status}" +
                                     (f"\n  " + "\n  ".join(issues) if issues else ""))
                return "## Results Validation\n\n" + "\n".join(checks)
            except Exception as e:
                return f"Validation error: {e}"

        return "Usage: visualize(job_id, action='summary'|'plot'|'list'|'validate', field='')"

    # ═══════════════════════════════════════════════════════════
    # 7. DEVELOPER (replaces 3 developer tools)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def developer(action: str, solver: str = "", keyword: str = "") -> str:
        """Developer tools: architecture, source files, capabilities matrix.

        Args:
            action: 'architecture' (extension points), 'files' (source listing),
                    'capabilities' (full matrix)
            solver: Backend name
            keyword: File pattern for 'files' action
        """
        if action == "architecture" and solver:
            from tools.developer import _SOURCE_LOCATIONS
            info = _SOURCE_LOCATIONS.get(solver, {})
            if not info:
                return f"Unknown solver: {solver}"
            return json.dumps(info, indent=2)

        elif action == "capabilities":
            lines = []
            for b in available_backends():
                physics = [p.name for p in b.supported_physics()]
                lines.append(f"**{b.display_name()}**: {', '.join(physics)}")
            return "\n".join(lines)

        elif action == "files" and solver:
            # Check if solver has a source root set via env var
            from tools.developer import _SOURCE_LOCATIONS
            info = _SOURCE_LOCATIONS.get(solver, {})
            source_root = info.get("root", "")
            source_env = info.get("source_env_var", "")

            # If keyword starts with "src/" or similar, search the solver source tree
            if keyword and source_root and Path(source_root).is_dir():
                base = Path(source_root)
                pattern = keyword
                files = sorted(base.rglob(pattern))[:30]
                if files:
                    return "\n".join(f"- {f.relative_to(base)} ({f.stat().st_size}b)" for f in files)

            # Default: search the MCP backend files
            base = Path(__file__).resolve().parents[1] / "backends" / solver
            if not base.exists():
                hint = f"\nTo browse {solver} source code, set {source_env} in .claude/settings.json" if source_env else ""
                return f"No source directory for {solver}{hint}"
            pattern = keyword or "*.py"
            files = sorted(base.rglob(pattern))
            result = "\n".join(f"- {f.relative_to(base)} ({f.stat().st_size}b)" for f in files[:20])
            if source_env and not (source_root and Path(source_root).is_dir()):
                result += f"\n\nNote: Set {source_env} env var to browse the full {solver} source tree"
            return result

        return "Usage: developer(action='architecture'|'capabilities'|'files', solver='')"

    # ═══════════════════════════════════════════════════════════
    # 8. PREPARE (meta-tool: knowledge + examples + template in one call)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def prepare_simulation(solver: str, physics: str) -> str:
        """Prepare everything needed to set up a simulation — in ONE call.

        Returns: knowledge + real test file examples + generated template.
        This eliminates 3 separate tool calls before every simulation.

        Supports fuzzy matching: e.g. 'magnetostatics' finds 'maxwell',
        'thermal' finds 'heat', 'elasticity' finds 'linear_elasticity'.

        Args:
            solver: Backend name (e.g. 'fourc', 'fenics', 'ngsolve')
            physics: Physics type (e.g. 'poisson', 'particle_pd', 'navier_stokes',
                     'magnetostatics', 'thermal', 'elasticity')
        """
        parts = []

        backend = get_backend(solver)
        if not backend:
            return f"Unknown solver: {solver}"

        # Fuzzy match: find the best matching physics name
        matched_physics = _fuzzy_match_physics(backend, physics)
        if matched_physics != physics:
            parts.append(f"*Note: '{physics}' matched to '{matched_physics}'*\n")

        # 0. Also available on — show which other backends support this physics (informational)
        alternatives = _list_alternative_solvers(solver, matched_physics)
        if alternatives:
            parts.append("## Also available on\n" + alternatives + "\n")

        # 1. Knowledge
        k = backend.get_knowledge(matched_physics)
        if k:
            parts.append("## Knowledge\n```json\n" + json.dumps(k, indent=2, default=str)[:3000] + "\n```\n")

        # 1b. General input-format pitfalls (ExodusII IDs, FUNCT syntax, etc.)
        # These apply to ALL physics in this solver, not just the current one
        general_k = backend.get_knowledge("input_format")
        if isinstance(general_k, dict):
            gp = general_k.get("general_pitfalls")
            if gp:
                pitfall_text = "\n".join(f"- {p}" for p in gp)
                parts.append(f"## General Input Pitfalls\n{pitfall_text}\n")

        # 2. Real test file examples
        from tools.knowledge import _find_reference_test_files
        ref = _find_reference_test_files(solver, matched_physics)
        if ref:
            parts.append(ref)

        # 3. Generated template
        for p in backend.supported_physics():
            if p.name == matched_physics and p.template_variants:
                try:
                    content = backend.generate_input(matched_physics, p.template_variants[0], {})
                    fmt = backend.input_format().value
                    parts.append(f"## Template ({p.template_variants[0]})\n```{fmt}\n{content[:3000]}\n```\n")
                except Exception:
                    pass
                break

        if not parts:
            # List available physics as hint
            avail = [p.name for p in backend.supported_physics()]
            return f"No information found for '{physics}' in {solver}. Available physics: {', '.join(avail)}"

        return f"# Preparation for {matched_physics} on {solver}\n\n" + "\n---\n".join(parts)

    # ═══════════════════════════════════════════════════════════
    # 9. TRANSFER FIELD (keep — needed for coupling)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    async def transfer_field(
        source_vtu: str, field_name: str,
        interface_coord: float, interface_axis: int = 0,
        target_format: str = "json", output_path: str = "",
    ) -> str:
        """Extract field values at an interface from a VTU file for cross-solver transfer.

        Args:
            source_vtu: Path to VTU file
            field_name: Field to extract (e.g. 'temperature')
            interface_coord: Coordinate of the interface plane
            interface_axis: Axis perpendicular to interface (0=x, 1=y, 2=z)
            target_format: 'json', 'fenics', '4c_neumann'
            output_path: Where to save (auto if empty)
        """
        from tools.coupling import register_coupling_tools
        # Delegate to original implementation
        from core.field_transfer import extract_interface_from_vtu
        import numpy as np

        vtu_path = Path(source_vtu)
        if not vtu_path.exists():
            return f"VTU file not found: {source_vtu}"

        try:
            iface = extract_interface_from_vtu(vtu_path, field_name, interface_coord, interface_axis)
        except Exception as e:
            return f"Error: {e}"

        vals = iface.values
        return (
            f"## Field Transfer: {field_name}\n"
            f"- Interface: {'xyz'[interface_axis]}={interface_coord}\n"
            f"- Nodes: {len(iface.coordinates)}\n"
            f"- Values: [{vals.min():.6e}, {vals.max():.6e}], mean={vals.mean():.6e}\n"
        )

    # ═══════════════════════════════════════════════════════════
    # 10. MESH (keep — needed for Gmsh)
    # ═══════════════════════════════════════════════════════════

    @mcp.tool()
    def generate_mesh(geometry: str, mesh_size: float = 0.1,
                      output_dir: str = "") -> str:
        """Generate a mesh using Gmsh for non-trivial geometries.

        Args:
            geometry: 'l_domain', 'plate_with_hole', 'channel_cylinder', or custom
            mesh_size: Target element size
            output_dir: Where to save (auto if empty)
        """
        from tools.mesh_generation import register_mesh_tools
        # Delegate to original
        try:
            from tools.mesh_generation import _generate_l_domain_2d, _generate_plate_with_hole_2d, _generate_channel_cylinder_2d
            generators = {
                "l_domain": _generate_l_domain_2d,
                "plate_with_hole": _generate_plate_with_hole_2d,
                "channel_cylinder": _generate_channel_cylinder_2d,
            }
            gen = generators.get(geometry)
            if gen:
                result = gen(mesh_size, output_dir or str(_OUTPUT_DIR / "meshes"))
                return result
            return f"Unknown geometry: {geometry}. Available: {list(generators.keys())}"
        except Exception as e:
            return f"Error: {e}"


# ═══════════════════════════════════════════════════════════════
# Helper functions for knowledge (copied from original tools)
# ═══════════════════════════════════════════════════════════════

def _get_coupling_knowledge():
    """Return coupling knowledge string."""
    try:
        from tools.knowledge import register_knowledge_tools
        from mcp.server.fastmcp import FastMCP
        mcp = FastMCP("tmp")
        captured = {}
        orig = mcp.tool
        def cap(*a, **kw):
            d = orig(*a, **kw)
            def w(fn):
                r = d(fn)
                captured[fn.__name__] = fn
                return r
            return w
        mcp.tool = cap
        register_knowledge_tools(mcp)
        if "get_coupling_knowledge" in captured:
            return captured["get_coupling_knowledge"]()
    except Exception:
        pass
    return "Coupling knowledge not available"

def _get_tsi_knowledge():
    try:
        from tools.knowledge import register_knowledge_tools
        from mcp.server.fastmcp import FastMCP
        mcp = FastMCP("tmp")
        captured = {}
        orig = mcp.tool
        def cap(*a, **kw):
            d = orig(*a, **kw)
            def w(fn):
                r = d(fn)
                captured[fn.__name__] = fn
                return r
            return w
        mcp.tool = cap
        register_knowledge_tools(mcp)
        if "get_tsi_knowledge" in captured:
            return captured["get_tsi_knowledge"]()
    except Exception:
        pass
    return "TSI knowledge not available"

def _get_precice_knowledge():
    try:
        from tools.knowledge import register_knowledge_tools
        from mcp.server.fastmcp import FastMCP
        mcp = FastMCP("tmp")
        captured = {}
        orig = mcp.tool
        def cap(*a, **kw):
            d = orig(*a, **kw)
            def w(fn):
                r = d(fn)
                captured[fn.__name__] = fn
                return r
            return w
        mcp.tool = cap
        register_knowledge_tools(mcp)
        if "get_precice_knowledge" in captured:
            return captured["get_precice_knowledge"]()
    except Exception:
        pass
    return "preCICE knowledge not available"
