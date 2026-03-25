"""
MCP tools for post-processing and visualization across backends.

Uses the unified post-processing module for consistent handling of all formats.
"""

import json
from pathlib import Path
from mcp.server.fastmcp import FastMCP
from tools.simulation import _jobs
from core.registry import get_backend


def register_visualization_tools(mcp: FastMCP):

    @mcp.tool()
    def post_process(job_id: str) -> str:
        """Extract field statistics from simulation results.

        Reads VTU/XDMF output and returns min/max/mean for each field,
        including QA checks for NaN/Inf and mesh quality.

        Args:
            job_id: The job ID from run_simulation.
        """
        job = _jobs.get(job_id)
        if not job:
            return f"Unknown job: {job_id}"
        if job.status != "completed":
            return f"Job {job_id} has status '{job.status}', cannot post-process."

        backend = get_backend(job.backend_name)
        if not backend:
            return f"Backend {job.backend_name} not found."

        result_files = backend.get_result_files(job)
        if not result_files:
            return "No output files found."

        vtu_files = [f for f in result_files if f.suffix == ".vtu"]
        xdmf_files = [f for f in result_files if f.suffix == ".xdmf"]
        files_to_process = vtu_files if vtu_files else xdmf_files

        if not files_to_process:
            return f"No processable output files. Found: {[f.name for f in result_files]}"

        from core.post_processing import post_process_file

        results = {}
        for fp in files_to_process:
            try:
                pp = post_process_file(fp, plot_dir=job.work_dir, plot_fields=False)
                results[fp.name] = pp.to_dict()
            except Exception as e:
                results[fp.name] = {"error": str(e)}

        return json.dumps(results, indent=2)

    @mcp.tool()
    def plot_result(job_id: str, field: str = "") -> str:
        """Generate a publication-quality visualization of simulation results.

        Args:
            job_id: The job ID from run_simulation.
            field: Field name to plot (e.g. 'u', 'temperature', 'displacement').
                   If empty, plots the first available field.
        """
        job = _jobs.get(job_id)
        if not job:
            return f"Unknown job: {job_id}"

        backend = get_backend(job.backend_name)
        if not backend:
            return f"Backend {job.backend_name} not found."

        result_files = backend.get_result_files(job)
        vtu_files = [f for f in result_files if f.suffix == ".vtu"]

        if not vtu_files:
            return "No VTU files found for plotting."

        try:
            import pyvista as pv
            pv.OFF_SCREEN = True
        except ImportError:
            return "PyVista not installed."

        from core.post_processing import plot_field, read_mesh, _detect_spatial_dim

        vtu = vtu_files[-1]
        mesh = read_mesh(vtu)

        if not field:
            all_fields = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
            if not all_fields:
                return "No fields found in output."
            field = all_fields[0]

        if field not in mesh.point_data and field not in mesh.cell_data:
            available = list(mesh.point_data.keys()) + list(mesh.cell_data.keys())
            return f"Field '{field}' not found. Available: {available}"

        plot_path = job.work_dir / f"plot_{field}.png"
        title = f"{backend.display_name()} — {field}"
        plot_field(mesh, field, plot_path, title=title)

        return f"Plot saved: {plot_path}"

    @mcp.tool()
    def compare_solvers(job_ids: str) -> str:
        """Compare results from multiple solver runs side by side.

        Args:
            job_ids: Comma-separated job IDs (e.g. 'abc123,def456,ghi789')
        """
        ids = [jid.strip() for jid in job_ids.split(",")]

        try:
            import pyvista as pv
            pv.OFF_SCREEN = True
        except ImportError:
            return "PyVista not installed."

        from core.post_processing import post_process_file

        results = {}
        for jid in ids:
            job = _jobs.get(jid)
            if not job:
                results[jid] = {"error": "Unknown job"}
                continue

            backend = get_backend(job.backend_name)
            if not backend:
                results[jid] = {"error": "Backend not found"}
                continue

            vtu_files = [f for f in backend.get_result_files(job) if f.suffix == ".vtu"]
            if not vtu_files:
                results[jid] = {"error": "No VTU output"}
                continue

            try:
                pp = post_process_file(vtu_files[-1], plot_fields=False)
                results[jid] = {
                    "solver": job.backend_name,
                    "display_name": backend.display_name(),
                    "status": job.status,
                    "elapsed": f"{job.elapsed:.2f}s" if job.elapsed else None,
                    "mesh": pp.mesh.to_dict(),
                    "fields": [f.to_dict() for f in pp.fields],
                }
            except Exception as e:
                results[jid] = {"solver": job.backend_name, "error": str(e)}

        return json.dumps(results, indent=2)

    @mcp.tool()
    def plot_all_fields(job_id: str) -> str:
        """Generate PNG plots for ALL fields in a simulation result.

        Args:
            job_id: The job ID from run_simulation.
        """
        job = _jobs.get(job_id)
        if not job:
            return f"Unknown job: {job_id}"

        backend = get_backend(job.backend_name)
        if not backend:
            return f"Backend {job.backend_name} not found."

        result_files = backend.get_result_files(job)
        vtu_files = [f for f in result_files if f.suffix == ".vtu"]

        if not vtu_files:
            return "No VTU files found."

        from core.post_processing import post_process_file

        all_plots = []
        for vtu in vtu_files:
            try:
                pp = post_process_file(vtu, plot_dir=job.work_dir, plot_fields=True)
                all_plots.extend(pp.plots)
            except Exception as e:
                all_plots.append(f"Error processing {vtu.name}: {e}")

        if not all_plots:
            return "No plots generated."

        return "Plots generated:\n" + "\n".join(f"- {p}" for p in all_plots)
