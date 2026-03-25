"""Kratos mesh generation utilities shared across generators."""


def generate_tri_mesh_mdpa(nx: int, ny: int, lx: float = 1.0, ly: float = 1.0) -> str:
    """Generate a Kratos .mdpa mesh string for a triangulated rectangle."""
    lines = ["Begin ModelPartData\nEnd ModelPartData\n",
             "Begin Properties 1\nEnd Properties\n"]

    # Nodes
    lines.append("Begin Nodes")
    nid = 1
    node_grid = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            x = i * lx / nx
            y = j * ly / ny
            node_grid[(i, j)] = nid
            lines.append(f"  {nid}  {x:.10f}  {y:.10f}  0.0")
            nid += 1
    lines.append("End Nodes\n")

    # Triangular elements (2 per quad cell)
    lines.append("Begin Elements Element2D3N")
    eid = 1
    for j in range(ny):
        for i in range(nx):
            n1 = node_grid[(i, j)]
            n2 = node_grid[(i+1, j)]
            n3 = node_grid[(i+1, j+1)]
            n4 = node_grid[(i, j+1)]
            lines.append(f"  {eid}  1  {n1}  {n2}  {n4}")
            eid += 1
            lines.append(f"  {eid}  1  {n2}  {n3}  {n4}")
            eid += 1
    lines.append("End Elements\n")

    # Boundary sub-model-parts
    # Left boundary (x=0)
    left_nodes = [node_grid[(0, j)] for j in range(ny + 1)]
    lines.append("Begin SubModelPart left")
    lines.append("  Begin SubModelPartNodes")
    for n in left_nodes:
        lines.append(f"    {n}")
    lines.append("  End SubModelPartNodes")
    lines.append("End SubModelPart\n")

    # Right boundary (x=lx)
    right_nodes = [node_grid[(nx, j)] for j in range(ny + 1)]
    lines.append("Begin SubModelPart right")
    lines.append("  Begin SubModelPartNodes")
    for n in right_nodes:
        lines.append(f"    {n}")
    lines.append("  End SubModelPartNodes")
    lines.append("End SubModelPart\n")

    # All boundary
    boundary = set()
    for i in range(nx + 1):
        boundary.add(node_grid[(i, 0)])
        boundary.add(node_grid[(i, ny)])
    for j in range(ny + 1):
        boundary.add(node_grid[(0, j)])
        boundary.add(node_grid[(nx, j)])
    lines.append("Begin SubModelPart boundary")
    lines.append("  Begin SubModelPartNodes")
    for n in sorted(boundary):
        lines.append(f"    {n}")
    lines.append("  End SubModelPartNodes")
    lines.append("End SubModelPart\n")

    return "\n".join(lines)
