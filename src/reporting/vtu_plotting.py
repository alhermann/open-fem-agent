"""
Visualize solution fields from 4C VTU output files using PyVista.

Creates publication-quality FEM field plots for inclusion in reports.
Includes mandatory QA checks on every generated plot.

PLOT QA RULES (non-negotiable):
- Color bars must NEVER overlap the mesh domain
- Titles must be above the domain, not overlapping
- 2D meshes: XY camera, domain in left ~75%, vertical cbar in right margin
- 3D meshes: isometric camera, horizontal cbar below the mesh
- Subplots: horizontal cbar below each subplot
- Color bars: thin, readable labels, not dominating the figure
"""

import os
os.environ["PYVISTA_OFF_SCREEN"] = "true"

import numpy as np
from pathlib import Path

import pyvista as pv
pv.global_theme.background = "white"
pv.global_theme.font.color = "black"


def _is_2d_mesh(mesh) -> bool:
    """Detect if mesh is 2D (constant z-coordinate)."""
    z_range = mesh.bounds[5] - mesh.bounds[4]
    return z_range < 1e-6


def _sbar_right(title: str, fmt: str = "%.3f", n_labels: int = 5) -> dict:
    """Vertical color bar in right margin. Use for 2D plots with shifted camera."""
    return {
        "title": title,
        "title_font_size": 16,
        "label_font_size": 14,
        "shadow": False,
        "n_labels": n_labels,
        "fmt": fmt,
        "position_x": 0.86,
        "position_y": 0.15,
        "width": 0.04,
        "height": 0.55,
        "vertical": True,
    }


def _sbar_bottom(title: str, fmt: str = "%.3f", n_labels: int = 5) -> dict:
    """Horizontal color bar at bottom. Use for 3D views and subplots."""
    return {
        "title": title,
        "title_font_size": 16,
        "label_font_size": 14,
        "shadow": False,
        "n_labels": n_labels,
        "fmt": fmt,
        "position_x": 0.15,
        "position_y": 0.05,
        "width": 0.7,
        "height": 0.05,
        "vertical": False,
    }


def _setup_2d_camera(plotter, mesh):
    """Set up 2D camera with domain shifted left to leave room for color bar.

    Handles meshes at any z-offset (e.g. z=999 in 4C fluid/FSI meshes).
    """
    plotter.view_xy()
    cx = (mesh.bounds[0] + mesh.bounds[1]) / 2
    cy = (mesh.bounds[2] + mesh.bounds[3]) / 2
    zval = mesh.bounds[4]  # actual z of the mesh (could be 0 or 999)
    dx = mesh.bounds[1] - mesh.bounds[0]
    dy = mesh.bounds[3] - mesh.bounds[2]
    # Shift camera right so domain sits in left ~75% of viewport
    cx_shifted = cx + dx * 0.12
    plotter.camera.position = (cx_shifted, cy, zval + 1.0)
    plotter.camera.focal_point = (cx_shifted, cy, zval)
    plotter.camera.up = (0, 1, 0)
    plotter.camera.parallel_projection = True
    plotter.camera.parallel_scale = max(dx, dy) * 0.55


def _setup_camera(plotter, mesh, force_2d=False):
    """Set camera appropriately for 2D or 3D meshes (for subplot use)."""
    if force_2d or _is_2d_mesh(mesh):
        plotter.view_xy()
        plotter.camera.tight(padding=0.1)
    else:
        plotter.view_isometric()


def _qa_check_plot(mesh, field_name: str, plot_path: Path) -> dict:
    """Mandatory QA check on a generated plot. Returns QA report dict."""
    qa = {"field": field_name, "file": str(plot_path), "checks": [], "passed": True}

    # 1. Check plot file exists and has reasonable size
    if plot_path.exists():
        size_kb = plot_path.stat().st_size / 1024
        if size_kb < 5:
            qa["checks"].append(("file_size", False, f"Plot file suspiciously small: {size_kb:.1f} KB"))
            qa["passed"] = False
        else:
            qa["checks"].append(("file_size", True, f"{size_kb:.1f} KB"))
    else:
        qa["checks"].append(("file_exists", False, "Plot file not created"))
        qa["passed"] = False
        return qa

    # 2. Check mesh has reasonable number of points/cells
    qa["checks"].append(("mesh_points", True, f"{mesh.n_points} points, {mesh.n_cells} cells"))
    if mesh.n_points < 3:
        qa["checks"].append(("mesh_quality", False, "Mesh has fewer than 3 points"))
        qa["passed"] = False

    # 3. Check field data sanity
    if field_name in mesh.point_data:
        data = mesh.point_data[field_name]
        if np.any(np.isnan(data)):
            qa["checks"].append(("nan_check", False, f"Field '{field_name}' contains NaN values"))
            qa["passed"] = False
        elif np.any(np.isinf(data)):
            qa["checks"].append(("inf_check", False, f"Field '{field_name}' contains Inf values"))
            qa["passed"] = False
        else:
            if data.ndim == 1:
                vmin, vmax = data.min(), data.max()
                qa["checks"].append(("field_range", True, f"[{vmin:.6g}, {vmax:.6g}]"))
            else:
                mag = np.linalg.norm(data, axis=1)
                qa["checks"].append(("field_range", True, f"magnitude [{mag.min():.6g}, {mag.max():.6g}]"))

    # 4. Check 2D vs 3D detection
    qa["checks"].append(("geometry", True, f"{'2D' if _is_2d_mesh(mesh) else '3D'} mesh"))

    return qa


def plot_scalar_field_2d(vtu_path: Path, field_name: str = "phi_1",
                          title: str = "", cbar_label: str = "",
                          save_path: Path | None = None) -> Path:
    """Plot a 2D scalar field from a VTU file as a filled contour."""
    mesh = pv.read(str(vtu_path))

    pl = pv.Plotter(off_screen=True, window_size=[1400, 1000])
    pl.add_mesh(
        mesh,
        scalars=field_name,
        cmap="viridis",
        show_edges=False,
        scalar_bar_args=_sbar_right(cbar_label or field_name),
    )
    pl.add_mesh(mesh, style="wireframe", color="gray", opacity=0.08, line_width=0.5)
    pl.add_title(title or f"Field: {field_name}", font_size=14)
    _setup_2d_camera(pl, mesh)

    out = save_path or (Path(vtu_path).parent.parent / f"plot_{field_name}.png")
    pl.screenshot(str(out), transparent_background=False, scale=2)
    pl.close()

    qa = _qa_check_plot(mesh, field_name, out)
    if not qa["passed"]:
        print(f"WARNING: Plot QA FAILED for {out}: {qa['checks']}")
    return out


def plot_displacement(vtu_path: Path, title: str = "",
                      save_path: Path | None = None,
                      warp_factor: float = 1.0) -> Path:
    """Plot displacement on a mesh with warped geometry. Auto-detects 2D/3D."""
    mesh = pv.read(str(vtu_path))

    if "displacement" not in mesh.point_data:
        print(f"WARNING: No 'displacement' field in {vtu_path}")
        return None

    disp = mesh.point_data["displacement"]
    disp_mag = np.linalg.norm(disp, axis=1)
    mesh.point_data["displacement_magnitude"] = disp_mag

    # Auto-scale warp factor if displacement is tiny relative to mesh size
    mesh_size = max(mesh.bounds[1] - mesh.bounds[0],
                    mesh.bounds[3] - mesh.bounds[2],
                    mesh.bounds[5] - mesh.bounds[4])
    max_disp = disp_mag.max()
    if max_disp > 0 and warp_factor == 1.0:
        auto_factor = 0.1 * mesh_size / max_disp
        if auto_factor > 10:
            warp_factor = auto_factor
            print(f"Auto-scaled warp factor to {warp_factor:.1f}")

    warped = mesh.warp_by_vector("displacement", factor=warp_factor)

    pl = pv.Plotter(off_screen=True, window_size=[1600, 900], shape=(1, 2))

    # Left: original mesh
    pl.subplot(0, 0)
    pl.add_mesh(mesh, color="steelblue", show_edges=True, edge_color="gray",
                opacity=0.7, line_width=0.5)
    pl.add_title("Undeformed", font_size=14)
    _setup_camera(pl, mesh)

    # Right: deformed mesh with horizontal cbar at bottom
    pl.subplot(0, 1)
    disp_fmt = "%.2e" if max_disp < 0.01 else "%.3f"
    pl.add_mesh(
        warped,
        scalars="displacement_magnitude",
        cmap="turbo",
        show_edges=True,
        edge_color="gray",
        line_width=0.5,
        scalar_bar_args=_sbar_bottom("|u|", fmt=disp_fmt),
    )
    warp_note = f" (warp x{warp_factor:.0f})" if warp_factor > 1.5 else ""
    pl.add_title(f"Deformed{warp_note}", font_size=14)
    _setup_camera(pl, warped)

    out = save_path or (Path(vtu_path).parent.parent / "plot_displacement.png")
    pl.screenshot(str(out), transparent_background=False, scale=2)
    pl.close()

    qa = _qa_check_plot(mesh, "displacement_magnitude", out)
    if not qa["passed"]:
        print(f"WARNING: Plot QA FAILED for {out}: {qa['checks']}")
    return out


# Keep old name as alias for backwards compatibility
plot_displacement_3d = plot_displacement


def plot_stress(vtu_path: Path, title: str = "",
                save_path: Path | None = None,
                warp_factor: float = 1.0) -> Path:
    """Plot von Mises stress on a deformed mesh. Auto-detects 2D/3D."""
    mesh = pv.read(str(vtu_path))

    if "nodal_cauchy_stresses_xyz" not in mesh.point_data:
        print(f"WARNING: No 'nodal_cauchy_stresses_xyz' field in {vtu_path}")
        return None

    stress = mesh.point_data["nodal_cauchy_stresses_xyz"]
    s11, s22, s33 = stress[:, 0], stress[:, 1], stress[:, 2]
    s12, s23, s13 = stress[:, 3], stress[:, 4], stress[:, 5]
    von_mises = np.sqrt(0.5 * ((s11-s22)**2 + (s22-s33)**2 + (s33-s11)**2
                                + 6*(s12**2 + s23**2 + s13**2)))
    mesh.point_data["von_mises"] = von_mises

    disp = mesh.point_data.get("displacement")
    if disp is not None:
        disp_mag = np.linalg.norm(disp, axis=1)
        mesh_size = max(mesh.bounds[1] - mesh.bounds[0],
                        mesh.bounds[3] - mesh.bounds[2],
                        mesh.bounds[5] - mesh.bounds[4])
        max_disp = disp_mag.max()
        if max_disp > 0 and warp_factor == 1.0:
            auto_factor = 0.1 * mesh_size / max_disp
            if auto_factor > 10:
                warp_factor = auto_factor
        warped = mesh.warp_by_vector("displacement", factor=warp_factor)
    else:
        warped = mesh

    # 3D view: horizontal cbar at bottom
    pl = pv.Plotter(off_screen=True, window_size=[1400, 900])
    pl.add_mesh(
        warped,
        scalars="von_mises",
        cmap="jet",
        show_edges=True,
        edge_color="gray",
        line_width=0.5,
        scalar_bar_args=_sbar_bottom("Von Mises [MPa]", fmt="%.1f"),
    )
    pl.add_title(title or "Von Mises Stress (deformed)", font_size=14)
    _setup_camera(pl, warped)

    out = save_path or (Path(vtu_path).parent.parent / "plot_vonmises.png")
    pl.screenshot(str(out), transparent_background=False, scale=2)
    pl.close()

    qa = _qa_check_plot(mesh, "von_mises", out)
    if not qa["passed"]:
        print(f"WARNING: Plot QA FAILED for {out}: {qa['checks']}")
    return out


# Keep old name as alias
plot_stress_3d = plot_stress


def plot_vector_field_2d(vtu_path: Path, field_name: str = "velocity",
                          title: str = "", save_path: Path | None = None) -> Path:
    """Plot a 2D vector field (e.g. velocity) with arrows and magnitude coloring."""
    mesh = pv.read(str(vtu_path))

    if field_name not in mesh.point_data:
        for name in mesh.point_data:
            if field_name.lower() in name.lower():
                field_name = name
                break
        else:
            print(f"WARNING: No field matching '{field_name}' in {vtu_path}")
            return None

    data = mesh.point_data[field_name]
    mag = np.linalg.norm(data, axis=1)
    mesh.point_data[f"{field_name}_magnitude"] = mag

    pl = pv.Plotter(off_screen=True, window_size=[1400, 1000])
    pl.add_mesh(
        mesh,
        scalars=f"{field_name}_magnitude",
        cmap="coolwarm",
        show_edges=False,
        scalar_bar_args=_sbar_right(f"|{field_name}|", fmt="%.4f"),
    )
    pl.add_mesh(mesh, style="wireframe", color="gray", opacity=0.05, line_width=0.5)
    pl.add_title(title or f"|{field_name}|", font_size=14)
    _setup_2d_camera(pl, mesh)

    out = save_path or (Path(vtu_path).parent.parent / f"plot_{field_name}.png")
    pl.screenshot(str(out), transparent_background=False, scale=2)
    pl.close()

    qa = _qa_check_plot(mesh, f"{field_name}_magnitude", out)
    if not qa["passed"]:
        print(f"WARNING: Plot QA FAILED for {out}: {qa['checks']}")
    return out


def plot_beam_displacement(vtu_path: Path, title: str = "",
                            save_path: Path | None = None) -> Path:
    """Plot beam element displacement (line elements in 3D space)."""
    mesh = pv.read(str(vtu_path))

    if "displacement" not in mesh.point_data:
        print(f"WARNING: No 'displacement' field in {vtu_path}")
        return None

    disp = mesh.point_data["displacement"]
    if disp.shape[1] > 3:
        disp_xyz = disp[:, :3]
    else:
        disp_xyz = disp
    disp_mag = np.linalg.norm(disp_xyz, axis=1)
    mesh.point_data["disp_mag"] = disp_mag
    mesh.point_data["disp_xyz"] = disp_xyz

    mesh_size = max(mesh.bounds[1] - mesh.bounds[0],
                    mesh.bounds[3] - mesh.bounds[2],
                    mesh.bounds[5] - mesh.bounds[4])
    max_disp = disp_mag.max()
    warp_factor = 1.0
    if max_disp > 0:
        warp_factor = max(1.0, 0.15 * mesh_size / max_disp)

    warped = mesh.copy()
    pts = warped.points.copy()
    pts += disp_xyz * warp_factor
    warped.points = pts

    pl = pv.Plotter(off_screen=True, window_size=[1600, 600], shape=(1, 2))

    pl.subplot(0, 0)
    pl.add_mesh(mesh, color="steelblue", line_width=4)
    pl.add_title("Undeformed", font_size=14)
    _setup_camera(pl, mesh)

    pl.subplot(0, 1)
    beam_fmt = "%.2e" if max_disp < 0.01 else "%.3f"
    pl.add_mesh(
        warped,
        scalars="disp_mag",
        cmap="turbo",
        line_width=4,
        scalar_bar_args=_sbar_bottom("|u|", fmt=beam_fmt),
    )
    warp_note = f" (warp x{warp_factor:.0f})" if warp_factor > 1.5 else ""
    pl.add_title(f"Deformed{warp_note}", font_size=14)
    _setup_camera(pl, warped)

    out = save_path or (Path(vtu_path).parent.parent / "plot_beam_displacement.png")
    pl.screenshot(str(out), transparent_background=False, scale=2)
    pl.close()

    qa = _qa_check_plot(mesh, "disp_mag", out)
    if not qa["passed"]:
        print(f"WARNING: Plot QA FAILED for {out}: {qa['checks']}")
    return out


def qa_check_vtu(vtu_path: Path) -> dict:
    """Standalone QA check on a VTU file without generating a plot."""
    mesh = pv.read(str(vtu_path))
    report = {
        "file": str(vtu_path),
        "n_points": mesh.n_points,
        "n_cells": mesh.n_cells,
        "bounds": list(mesh.bounds),
        "is_2d": _is_2d_mesh(mesh),
        "fields": {},
        "issues": [],
    }

    for name in mesh.point_data:
        data = mesh.point_data[name]
        info = {"shape": list(data.shape)}
        if np.any(np.isnan(data)):
            info["has_nan"] = True
            report["issues"].append(f"Field '{name}' contains NaN")
        if np.any(np.isinf(data)):
            info["has_inf"] = True
            report["issues"].append(f"Field '{name}' contains Inf")
        if data.ndim == 1:
            info["min"] = float(data.min())
            info["max"] = float(data.max())
            info["mean"] = float(data.mean())
        else:
            mag = np.linalg.norm(data, axis=1)
            info["mag_min"] = float(mag.min())
            info["mag_max"] = float(mag.max())
            info["mag_mean"] = float(mag.mean())
        report["fields"][name] = info

    if mesh.n_points < 3:
        report["issues"].append("Mesh has fewer than 3 points")
    if mesh.n_cells < 1:
        report["issues"].append("Mesh has no cells")

    report["passed"] = len(report["issues"]) == 0
    return report
