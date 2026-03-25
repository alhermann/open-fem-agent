"""scikit-fem Poisson equation generators and knowledge."""
import math


def _poisson_2d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Poisson -Δu = f on [0,1]² with Q1 quads, u=0 on boundary."""
    nx = params.get("nx", 32)
    f_val = params.get("f", 1.0)
    return f'''\
"""Poisson -Δu = {f_val} on [0,1]², Q1 quads, u=0 on boundary — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, unit_load
import numpy as np
import json

# Mesh: structured quad mesh on [0,1]²
m = MeshQuad.init_tensor(np.linspace(0, 1, {nx + 1}), np.linspace(0, 1, {nx + 1}))
e = ElementQuad1()
ib = Basis(m, e)

# Assembly
K = laplace.assemble(ib)
f = ib.zeros()
f += {f_val} * unit_load.assemble(ib)

# Dirichlet BC: u=0 on all boundaries
D = ib.get_dofs().flatten()
u = solve(*condense(K, f, D=D))

max_val = u.max()
print(f"max(u) = {{max_val:.10f}}")
print(f"DOFs: {{K.shape[0]}}")
print(f"Elements: {{m.nelements}}")

# VTK output via meshio
import meshio
cells = [("quad", m.t.T)]
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])]) if m.p.shape[0] == 2 else m.p.T
mio = meshio.Mesh(points, cells, point_data={{"phi": u}})
mio.write("result.vtu")

summary = {{
    "max_value": float(max_val),
    "n_dofs": int(K.shape[0]),
    "n_elements": int(m.nelements),
    "element_type": "Q1 quad",
}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
print("Poisson solve complete.")
'''


def _poisson_2d_tri(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Poisson -Δu = f on [0,1]² with P1 triangles."""
    nx = params.get("nx", 32)
    f_val = params.get("f", 1.0)
    refine_level = max(int(math.log2(nx)), 3) if nx > 1 else 3
    return f'''\
"""Poisson -Δu = {f_val} on [0,1]², P1 triangles — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, unit_load
import numpy as np
import json

m = MeshTri.init_symmetric().refined({refine_level})
e = ElementTriP1()
ib = Basis(m, e)

K = laplace.assemble(ib)
f = ib.zeros()
f += {f_val} * unit_load.assemble(ib)

D = ib.get_dofs().flatten()
u = solve(*condense(K, f, D=D))

max_val = u.max()
print(f"max(u) = {{max_val:.10f}}")

import meshio
cells = [("quad", m.t.T)]
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])]) if m.p.shape[0] == 2 else m.p.T
mio = meshio.Mesh(points, cells, point_data={{"phi": u}})
mio.write("result.vtu")

summary = {{"max_value": float(max_val), "n_dofs": int(K.shape[0]), "element_type": "P1 tri"}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


def _poisson_3d(params: dict) -> str:
    """FORMAT TEMPLATE — values are defaults, determine appropriate values for your specific problem.

    Poisson on [0,1]³ with hex elements."""
    nx = params.get("nx", 8)
    f_val = params.get("f", 1.0)
    return f'''\
"""Poisson -Δu = {f_val} on [0,1]³, Hex1 — scikit-fem"""
from skfem import *
from skfem.models.poisson import laplace, unit_load
import numpy as np
import json

m = MeshHex.init_tensor(np.linspace(0,1,{nx+1}), np.linspace(0,1,{nx+1}), np.linspace(0,1,{nx+1}))
e = ElementHex1()
ib = Basis(m, e)

K = laplace.assemble(ib)
f = ib.zeros()
f += {f_val} * unit_load.assemble(ib)

D = ib.get_dofs().flatten()
u = solve(*condense(K, f, D=D))
print(f"3D Poisson max(u) = {{u.max():.10f}}")

import meshio
cells = [("quad", m.t.T)]
points = np.column_stack([m.p.T, np.zeros(m.p.shape[1])]) if m.p.shape[0] == 2 else m.p.T
mio = meshio.Mesh(points, cells, point_data={{"phi": u}})
mio.write("result.vtu")

summary = {{"max_value": float(u.max()), "n_dofs": int(K.shape[0])}}
with open("results_summary.json", "w") as _f:
    json.dump(summary, _f, indent=2)
'''


KNOWLEDGE = {
    "poisson": {
        "description": "Poisson with scikit-fem (pure Python assembly)",
        "solver": "scipy.sparse.linalg.spsolve (direct) or eigsh (eigenvalue)",
        "elements": "ElementTriP1/P2/P3, ElementQuad1/2, ElementTetP1/P2, ElementHex1/2",
        "built_in_forms": "laplace, unit_load (from skfem.models.poisson)",
        "pitfalls": [
            "scikit-fem is an ASSEMBLY library — you control the solve",
            "Boundary DOFs: ib.get_dofs().flatten() for all, ib.get_dofs('left') for named",
            "VTU output: meshio.Mesh(points, cells, point_data).write('result.vtu')",
            "solve() and condense() handle Dirichlet elimination",
            "For v12+: to_meshio removed — use meshio.Mesh() directly",
        ],
    },
}

GENERATORS = {
    "poisson_2d": _poisson_2d,
    "poisson_2d_tri": _poisson_2d_tri,
    "poisson_3d": _poisson_3d,
}
