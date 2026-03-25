#!/usr/bin/env python3
"""
Generator script for 4C FSI simulation: flow around an elastic beam in a 2D channel.

Problem: Turek-Hron FSI benchmark variant
- 2D channel [0, 2.5] x [0, 0.41]
- Elastic beam [0.59, 0.61] x [0, 0.35], fixed at bottom wall
- Parabolic inflow with cosine ramp over 2s
- NeoHooke structure, incompressible Navier-Stokes fluid

Mesh: 4 structured QUAD4 blocks with FSI interface node duplication.
The FT block uses x-coordinates that match the FL+BEAM+FR bottom row
to ensure conforming fluid-fluid interfaces.
"""

import numpy as np
import meshio
import os

# =============================================================================
# Geometry parameters
# =============================================================================
L = 2.5       # channel length
H = 0.41      # channel height
x1 = 0.59     # beam left edge
x2 = 0.61     # beam right edge
hb = 0.35     # beam height

# Mesh resolution per block
nx_L = 20     # FL: fluid left, x-direction
nx_S = 2      # BEAM: structure, x-direction
nx_R = 50     # FR: fluid right, x-direction
ny_B = 25     # FL/BEAM/FR: y-direction (bottom row)
nx_T = nx_L + nx_S + nx_R  # FT: fluid top, x-direction (= 72)
ny_T = 5      # FT: fluid top, y-direction

# =============================================================================
# Helper: generate structured QUAD4 block from explicit x/y arrays
# =============================================================================
def make_block_from_arrays(xs, ys):
    """
    Create a structured QUAD4 mesh block from explicit 1D coordinate arrays.
    Returns: points (ny+1)*(nx+1) x 2, quads nx*ny x 4
    Node ordering: row-major, y varies slowest.
      node(i,j) = j*(nx+1) + i,  i=0..nx, j=0..ny
    """
    nx = len(xs) - 1
    ny = len(ys) - 1
    xx, yy = np.meshgrid(xs, ys)  # shape (ny+1, nx+1)
    points = np.column_stack([xx.ravel(), yy.ravel()])  # (ny+1)*(nx+1) x 2

    quads = []
    for j in range(ny):
        for i in range(nx):
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n1 + (nx + 1)
            n3 = n0 + (nx + 1)
            quads.append([n0, n1, n2, n3])
    quads = np.array(quads, dtype=int)
    return points, quads, nx, ny


# =============================================================================
# Build x/y coordinate arrays for each block
# =============================================================================
fl_xs = np.linspace(0, x1, nx_L + 1)
s_xs = np.linspace(x1, x2, nx_S + 1)
fr_xs = np.linspace(x2, L, nx_R + 1)
bottom_ys = np.linspace(0, hb, ny_B + 1)
top_ys = np.linspace(hb, H, ny_T + 1)

# FT x-coordinates: concatenation of FL, BEAM, FR x-coordinates (unique, sorted)
# This ensures FT bottom row aligns exactly with FL/BEAM/FR top rows
ft_xs = np.concatenate([fl_xs, s_xs[1:], fr_xs[1:]])  # remove duplicate x1, x2
assert len(ft_xs) == nx_T + 1, f"ft_xs has {len(ft_xs)} points, expected {nx_T+1}"

# =============================================================================
# Build the four blocks
# =============================================================================
fl_pts, fl_quads, _, _ = make_block_from_arrays(fl_xs, bottom_ys)
s_pts, s_quads, _, _ = make_block_from_arrays(s_xs, bottom_ys)
fr_pts, fr_quads, _, _ = make_block_from_arrays(fr_xs, bottom_ys)
ft_pts, ft_quads, _, _ = make_block_from_arrays(ft_xs, top_ys)

print("=" * 60)
print("BLOCK SIZES (before merging)")
print(f"  FL:   {fl_pts.shape[0]} nodes, {fl_quads.shape[0]} elements")
print(f"  BEAM: {s_pts.shape[0]} nodes, {s_quads.shape[0]} elements")
print(f"  FR:   {fr_pts.shape[0]} nodes, {fr_quads.shape[0]} elements")
print(f"  FT:   {ft_pts.shape[0]} nodes, {ft_quads.shape[0]} elements")

# =============================================================================
# Global node assembly with merging rules:
#   - FL, FR, FT share nodes at common fluid-fluid boundaries (merged)
#   - BEAM has SEPARATE nodes from FL/FR/FT (FSI interface — NOT shared)
#
# Strategy: Build global arrays by adding blocks and tracking merges.
# FT bottom row aligns with FL top row (i=0..nx_L) and FR top row
# (i=nx_L+nx_S..nx_T), but NOT with BEAM top row (i=nx_L..nx_L+nx_S).
# The FT nodes at i=nx_L..nx_L+nx_S, j=0 are new fluid nodes above the beam.
# =============================================================================
ATOL = 1e-10  # tolerance for coordinate matching

# Start with FL nodes as the base
global_pts = list(fl_pts)  # list of [x,y] arrays
fl_gid = np.arange(len(fl_pts), dtype=int)  # global IDs for FL nodes
n_global = len(global_pts)

# --- Add BEAM nodes (all separate, no merging with fluid) ---
s_gid = np.arange(n_global, n_global + len(s_pts), dtype=int)
for p in s_pts:
    global_pts.append(p)
n_global += len(s_pts)

# --- Add FR nodes (all new — FR shares no boundary with FL directly) ---
fr_gid = np.arange(n_global, n_global + len(fr_pts), dtype=int)
for p in fr_pts:
    global_pts.append(p)
n_global += len(fr_pts)

# --- Add FT nodes with merging ---
# FT bottom row (j=0) has nx_T+1 nodes. Their x-coordinates match:
#   i=0..nx_L        -> FL top row (i=0..nx_L, j=ny_B): MERGE (fluid-fluid)
#   i=nx_L..nx_L+nx_S -> above beam: NEW fluid nodes (NOT merged with beam)
#   i=nx_L+nx_S..nx_T -> FR top row (i=0..nx_R, j=ny_B): MERGE (fluid-fluid)
# FT rows j>0 are all new.

ft_gid = np.full(len(ft_pts), -1, dtype=int)

for local_idx in range(len(ft_pts)):
    j_ft = local_idx // (nx_T + 1)  # row index in FT
    i_ft = local_idx % (nx_T + 1)   # column index in FT

    merged = False

    if j_ft == 0:  # bottom row of FT (y=hb)
        if i_ft <= nx_L:
            # Merge with FL top row node at (i=i_ft, j=ny_B)
            fl_local = ny_B * (nx_L + 1) + i_ft
            ft_gid[local_idx] = fl_gid[fl_local]
            merged = True
        elif i_ft >= nx_L + nx_S:
            # Merge with FR top row node at (i=i_ft-(nx_L+nx_S), j=ny_B)
            fr_i = i_ft - (nx_L + nx_S)
            fr_local = ny_B * (nx_R + 1) + fr_i
            ft_gid[local_idx] = fr_gid[fr_local]
            merged = True
        # else: i_ft in (nx_L, nx_L+nx_S) exclusive — above beam, new fluid node

    if not merged:
        ft_gid[local_idx] = n_global
        global_pts.append(ft_pts[local_idx])
        n_global += 1

global_pts = np.array(global_pts)  # (n_global, 2)

print(f"\nTotal global nodes: {n_global}")

# Verify FT merge counts
ft_merged_fl = sum(1 for idx in range(nx_T + 1) if idx <= nx_L)
ft_merged_fr = sum(1 for idx in range(nx_T + 1) if idx >= nx_L + nx_S)
ft_new_above_beam = sum(1 for idx in range(nx_T + 1) if nx_L < idx < nx_L + nx_S)
print(f"FT bottom row merges: {ft_merged_fl} with FL, {ft_merged_fr} with FR, "
      f"{ft_new_above_beam} new above beam")

# =============================================================================
# Build global element connectivity
# =============================================================================
# Structure elements (block 1 in ExodusII)
s_elems = s_gid[s_quads]  # shape (n_s_elem, 4)

# Fluid elements (block 2 in ExodusII): FL + FR + FT
fl_elems = fl_gid[fl_quads]
fr_elems = fr_gid[fr_quads]
ft_elems = ft_gid[ft_quads]
f_elems = np.vstack([fl_elems, fr_elems, ft_elems])

n_s_elem = s_elems.shape[0]
n_f_elem = f_elems.shape[0]
print(f"Structure elements (block 1): {n_s_elem}")
print(f"Fluid elements (block 2): {n_f_elem}")
print(f"  FL: {fl_elems.shape[0]}, FR: {fr_elems.shape[0]}, FT: {ft_elems.shape[0]}")

# =============================================================================
# Identify boundary nodes for DLINE conditions (using 0-based internal IDs)
# =============================================================================

# Helper: get global node ID for each block's grid position
def fl_node(i, j):
    """FL node at grid position (i,j), returns global ID."""
    return int(fl_gid[j * (nx_L + 1) + i])

def s_node(i, j):
    """BEAM node at grid position (i,j), returns global ID."""
    return int(s_gid[j * (nx_S + 1) + i])

def fr_node(i, j):
    """FR node at grid position (i,j), returns global ID."""
    return int(fr_gid[j * (nx_R + 1) + i])

def ft_node(i, j):
    """FT node at grid position (i,j), returns global ID."""
    return int(ft_gid[j * (nx_T + 1) + i])


# --- DLINE 1: Structure fixed bottom (y=0, beam bottom row j=0) ---
dline1_nodes = sorted(set(s_node(i, 0) for i in range(nx_S + 1)))

# --- DLINE 2: Structure FSI interface (left + top + right surfaces of beam) ---
dline2_set = set()
# Left surface: i=0, j=0..ny_B
for j in range(ny_B + 1):
    dline2_set.add(s_node(0, j))
# Top surface: j=ny_B, i=0..nx_S
for i in range(nx_S + 1):
    dline2_set.add(s_node(i, ny_B))
# Right surface: i=nx_S, j=0..ny_B
for j in range(ny_B + 1):
    dline2_set.add(s_node(nx_S, j))
dline2_nodes = sorted(dline2_set)

# --- DLINE 3: Fluid FSI interface (FL right edge + FT bottom center + FR left edge) ---
dline3_set = set()
# FL right edge: i=nx_L, j=0..ny_B
for j in range(ny_B + 1):
    dline3_set.add(fl_node(nx_L, j))
# FT bottom row above beam: i = nx_L+1 to nx_L+nx_S-1 (strictly between FL and FR)
# Plus the corners at i=nx_L and i=nx_L+nx_S which are already FL/FR top nodes
# Actually: the FT bottom nodes above the beam are at i=nx_L..nx_L+nx_S
# But i=nx_L is merged with FL(nx_L, ny_B) and i=nx_L+nx_S is merged with FR(0, ny_B)
# So we need to add ALL FT bottom nodes from i=nx_L to i=nx_L+nx_S
for i in range(nx_L, nx_L + nx_S + 1):
    dline3_set.add(ft_node(i, 0))
# FR left edge: i=0, j=0..ny_B
for j in range(ny_B + 1):
    dline3_set.add(fr_node(0, j))
dline3_nodes = sorted(dline3_set)

# --- DLINE 4: Fluid inflow (x=0) ---
dline4_set = set()
# FL left edge: i=0, j=0..ny_B
for j in range(ny_B + 1):
    dline4_set.add(fl_node(0, j))
# FT left edge: i=0, j=0..ny_T (j=0 merged with FL top-left)
for j in range(ny_T + 1):
    dline4_set.add(ft_node(0, j))
dline4_nodes = sorted(dline4_set)

# --- DLINE 5: Fluid outflow (x=L) ---
dline5_set = set()
# FR right edge: i=nx_R, j=0..ny_B
for j in range(ny_B + 1):
    dline5_set.add(fr_node(nx_R, j))
# FT right edge: i=nx_T, j=0..ny_T (j=0 merged with FR top-right)
for j in range(ny_T + 1):
    dline5_set.add(ft_node(nx_T, j))
dline5_nodes = sorted(dline5_set)

# --- DLINE 6: Fluid bottom wall (y=0, excluding beam base) ---
dline6_set = set()
# FL bottom: i=0..nx_L, j=0
for i in range(nx_L + 1):
    dline6_set.add(fl_node(i, 0))
# FR bottom: i=0..nx_R, j=0
for i in range(nx_R + 1):
    dline6_set.add(fr_node(i, 0))
dline6_nodes = sorted(dline6_set)

# --- DLINE 7: Fluid top wall (y=H) ---
dline7_set = set()
# FT top row: j=ny_T, i=0..nx_T
for i in range(nx_T + 1):
    dline7_set.add(ft_node(i, ny_T))
dline7_nodes = sorted(dline7_set)

# --- Beam tip node (DNODE 1): center of beam top edge ---
tip_local_i = nx_S // 2  # nx_S=2 -> i=1
tip_gid = s_node(tip_local_i, ny_B)
tip_nid = tip_gid + 1  # 1-based for 4C
tip_coords = global_pts[tip_gid]

print(f"\nBeam tip node: global_0based={tip_gid}, 1based_nid={tip_nid}, "
      f"coords=({tip_coords[0]:.4f}, {tip_coords[1]:.4f})")

print(f"\nDLINE node counts:")
print(f"  DLINE 1 (struct fixed bottom): {len(dline1_nodes)}")
print(f"  DLINE 2 (struct FSI interface): {len(dline2_nodes)}")
print(f"  DLINE 3 (fluid FSI interface):  {len(dline3_nodes)}")
print(f"  DLINE 4 (fluid inflow):         {len(dline4_nodes)}")
print(f"  DLINE 5 (fluid outflow):        {len(dline5_nodes)}")
print(f"  DLINE 6 (fluid bottom wall):    {len(dline6_nodes)}")
print(f"  DLINE 7 (fluid top wall):       {len(dline7_nodes)}")

# Verify FSI interface: structure and fluid nodes must be at same positions but different IDs
s_fsi_coords = set()
for nid in dline2_nodes:
    s_fsi_coords.add((round(global_pts[nid, 0], 10), round(global_pts[nid, 1], 10)))
f_fsi_coords = set()
for nid in dline3_nodes:
    f_fsi_coords.add((round(global_pts[nid, 0], 10), round(global_pts[nid, 1], 10)))
shared_coords = s_fsi_coords & f_fsi_coords
shared_ids = set(dline2_nodes) & set(dline3_nodes)
print(f"\nFSI interface verification:")
print(f"  Struct FSI positions: {len(s_fsi_coords)}")
print(f"  Fluid FSI positions:  {len(f_fsi_coords)}")
print(f"  Matching positions:   {len(shared_coords)}")
print(f"  Shared node IDs:      {len(shared_ids)} (MUST be 0 for proper FSI)")
assert len(shared_ids) == 0, "ERROR: Structure and fluid FSI nodes share IDs!"
assert len(shared_coords) == len(s_fsi_coords), \
    f"ERROR: Not all struct FSI positions have matching fluid positions! " \
    f"{len(shared_coords)} vs {len(s_fsi_coords)}"

# =============================================================================
# Write ExodusII mesh file
# =============================================================================
# Add z=0 coordinate for ExodusII (3D required)
pts_3d = np.column_stack([global_pts, np.zeros(n_global)])

# meshio cells: block 1 = structure, block 2 = fluid
cells = [
    meshio.CellBlock("quad", s_elems),
    meshio.CellBlock("quad", f_elems),
]

mesh = meshio.Mesh(points=pts_3d, cells=cells)
mesh_file = "mesh.e"
meshio.exodus.write(mesh_file, mesh)
print(f"\nWrote {mesh_file} with meshio")

# Patch block IDs: meshio writes 0-indexed, 4C needs 1-indexed
import netCDF4
ds = netCDF4.Dataset(mesh_file, "r+")
eb_prop = ds.variables["eb_prop1"]
old_ids = eb_prop[:].copy()
eb_prop[:] = old_ids + 1
print(f"Patched eb_prop1: {old_ids} -> {eb_prop[:]}")
ds.close()

# Verify
ds = netCDF4.Dataset(mesh_file, "r")
print(f"Verified eb_prop1: {ds.variables['eb_prop1'][:]}")
ds.close()

# =============================================================================
# Build DLINE-NODE TOPOLOGY and DNODE-NODE TOPOLOGY strings
# =============================================================================
# 4C uses 1-based node IDs: internal_gid + 1
dline_topo_lines = []
for nid in dline1_nodes:
    dline_topo_lines.append(f'"NODE {nid+1} DLINE 1"')
for nid in dline2_nodes:
    dline_topo_lines.append(f'"NODE {nid+1} DLINE 2"')
for nid in dline3_nodes:
    dline_topo_lines.append(f'"NODE {nid+1} DLINE 3"')
for nid in dline4_nodes:
    dline_topo_lines.append(f'"NODE {nid+1} DLINE 4"')
for nid in dline5_nodes:
    dline_topo_lines.append(f'"NODE {nid+1} DLINE 5"')
for nid in dline6_nodes:
    dline_topo_lines.append(f'"NODE {nid+1} DLINE 6"')
for nid in dline7_nodes:
    dline_topo_lines.append(f'"NODE {nid+1} DLINE 7"')

dline_topo_str = "\n".join(f"  - {line}" for line in dline_topo_lines)
dnode_topo_str = f'  - "NODE {tip_nid} DNODE 1"'

print(f"\nDLINE-NODE TOPOLOGY: {len(dline_topo_lines)} entries")

# =============================================================================
# Write 4C YAML input file
# =============================================================================
yaml_content = f"""\
TITLE:
  - "FSI beam in 2D channel - elastic beam benchmark"
PROBLEM SIZE:
  DIM: 2
PROBLEM TYPE:
  PROBLEMTYPE: Fluid_Structure_Interaction
IO:
  FILESTEPS: 1000
  VERBOSITY: "standard"
IO/RUNTIME VTK OUTPUT:
  OUTPUT_DATA_FORMAT: "binary"
  INTERVAL_STEPS: 5
IO/RUNTIME VTK OUTPUT/FLUID:
  OUTPUT_FLUID: true
  VELOCITY: true
  PRESSURE: true
IO/RUNTIME VTK OUTPUT/ALE:
  OUTPUT_ALE: true
  DISPLACEMENT: true
ALE DYNAMIC:
  ALE_TYPE: springs_spatial
  MAXITER: 4
  TOLRES: 1e-4
  TOLDISP: 1e-4
  RESULTSEVERY: 5
  LINEAR_SOLVER: 1
FLUID DYNAMIC:
  LINEAR_SOLVER: 2
  TIMEINTEGR: "Np_Gen_Alpha"
  GRIDVEL: BDF2
  ADAPTCONV: true
  ITEMAX: 50
FLUID DYNAMIC/NONLINEAR SOLVER TOLERANCES:
  TOL_VEL_RES: 1e-6
  TOL_VEL_INC: 1e-6
  TOL_PRES_RES: 1e-6
  TOL_PRES_INC: 1e-6
FLUID DYNAMIC/RESIDUAL-BASED STABILIZATION:
  CHARELELENGTH_PC: "root_of_volume"
FSI DYNAMIC:
  COUPALGO: iter_monolithicstructuresplit
  MAXTIME: 2.0
  NUMSTEP: 200
  TIMESTEP: 0.01
  SECONDORDER: true
  RESULTSEVERY: 5
FSI DYNAMIC/MONOLITHIC SOLVER:
  SHAPEDERIVATIVES: true
  LINEARBLOCKSOLVER: LinalgSolver
  LINEAR_SOLVER: 4
SOLVER 1:
  SOLVER: "UMFPACK"
  NAME: "ALE solver"
SOLVER 2:
  SOLVER: "UMFPACK"
  NAME: "Fluid solver"
SOLVER 3:
  SOLVER: "UMFPACK"
  NAME: "Structure solver"
SOLVER 4:
  SOLVER: "UMFPACK"
  NAME: "FSI solver"
STRUCTURAL DYNAMIC:
  INT_STRATEGY: "Standard"
  M_DAMP: 0.0
  K_DAMP: 0.01
  TOLDISP: 1e-8
  TOLRES: 1e-8
  PREDICT: "ConstDisVelAcc"
  LINEAR_SOLVER: 3
STRUCTURAL DYNAMIC/GENALPHA:
  BETA: 0.25
  GAMMA: 0.5
  ALPHA_M: 0.5
  ALPHA_F: 0.5
  RHO_INF: -1
MATERIALS:
  - MAT: 1
    MAT_fluid:
      DYNVISCOSITY: 0.01
      DENSITY: 1.0
  - MAT: 2
    MAT_ElastHyper:
      NUMMAT: 1
      MATIDS: [3]
      DENS: 10.0
  - MAT: 3
    ELAST_CoupNeoHooke:
      YOUNG: 5000.0
      NUE: 0.4
  - MAT: 4
    MAT_Struct_StVenantKirchhoff:
      YOUNG: 1.0
      NUE: 0.0
      DENS: 1.0
CLONING MATERIAL MAP:
  - SRC_FIELD: "fluid"
    SRC_MAT: 1
    TAR_FIELD: "ale"
    TAR_MAT: 4
FUNCT1:
  - SYMBOLIC_FUNCTION_OF_SPACE_TIME: "6*y*(0.41-y)/(0.41*0.41)*0.5*(1-cos(pi*t/2))"
STRUCTURE GEOMETRY:
  FILE: mesh.e
  ELEMENT_BLOCKS:
    - ID: 1
      WALL:
        QUAD4:
          MAT: 2
          KINEM: nonlinear
          EAS: none
          THICK: 1.0
          STRESS_STRAIN: plane_strain
          GP: [2, 2]
FLUID GEOMETRY:
  FILE: mesh.e
  SHOW_INFO: "summary"
  ELEMENT_BLOCKS:
    - ID: 2
      FLUID:
        QUAD4:
          MAT: 1
          NA: ALE
DNODE-NODE TOPOLOGY:
{dnode_topo_str}
DLINE-NODE TOPOLOGY:
{dline_topo_str}
DESIGN LINE DIRICH CONDITIONS:
  - E: 1
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0.0, 0.0]
    FUNCT: [0, 0]
  - E: 4
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [1.0, 0.0, 0.0]
    FUNCT: [1, 0, 0]
  - E: 6
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
  - E: 7
    NUMDOF: 3
    ONOFF: [1, 1, 0]
    VAL: [0.0, 0.0, 0.0]
    FUNCT: [0, 0, 0]
DESIGN LINE ALE DIRICH CONDITIONS:
  - E: 4
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0.0, 0.0]
    FUNCT: [0, 0]
  - E: 5
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0.0, 0.0]
    FUNCT: [0, 0]
  - E: 6
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0.0, 0.0]
    FUNCT: [0, 0]
  - E: 7
    NUMDOF: 2
    ONOFF: [1, 1]
    VAL: [0.0, 0.0]
    FUNCT: [0, 0]
DESIGN FSI COUPLING LINE CONDITIONS:
  - E: 2
    coupling_id: 1
  - E: 3
    coupling_id: 1
RESULT DESCRIPTION:
  - STRUCTURE:
      DIS: "structure"
      NODE: {tip_nid}
      QUANTITY: "dispx"
      VALUE: 0.0
      TOLERANCE: 1.0
  - STRUCTURE:
      DIS: "structure"
      NODE: {tip_nid}
      QUANTITY: "dispy"
      VALUE: 0.0
      TOLERANCE: 1.0
"""

yaml_file = "fsi_beam.4C.yaml"
with open(yaml_file, "w") as f:
    f.write(yaml_content)

print(f"\nWrote {yaml_file}")

# =============================================================================
# Final summary
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Mesh file:        {os.path.abspath(mesh_file)}")
print(f"YAML file:        {os.path.abspath(yaml_file)}")
print(f"Total nodes:      {n_global}")
print(f"Struct elements:  {n_s_elem} (block 1)")
print(f"Fluid elements:   {n_f_elem} (block 2)")
print(f"  FL elements:    {fl_elems.shape[0]}")
print(f"  FR elements:    {fr_elems.shape[0]}")
print(f"  FT elements:    {ft_elems.shape[0]}")
print(f"Beam tip node:    ID={tip_nid} (1-based), coords=({tip_coords[0]:.4f}, {tip_coords[1]:.4f})")
print(f"FSI struct nodes: {len(dline2_nodes)}")
print(f"FSI fluid nodes:  {len(dline3_nodes)}")
print(f"Shared FSI IDs:   {len(shared_ids)} (must be 0)")
print("=" * 60)
