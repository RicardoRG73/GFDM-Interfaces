""" Importing needed libraries """
import numpy as np
import matplotlib.pyplot as plt

# calfem-python
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

""" Creating geometry object """
geometry = cfg.Geometry()                   # geometry object

# points
geometry.point([-1,0])                      # 0
geometry.point([1,0])                       # 3
geometry.point([1,1])                       # 4
geometry.point([-1,1])                      # 7

# lines
left = 11                                   # marker for nodes on left boundary
right = 12                                  # marker for nodes on right boundary
bottom = 13
top = 14
geometry.spline([0,1], marker=bottom)      # 0
geometry.spline([1,2], marker=right)        # 1
geometry.spline([2,3], marker=top)      # 2
geometry.spline([3,0], marker=left)         # 3


# surfaces
mat0 = 100                                  # marker for nodes on material 1
geometry.surface([0,1,2,3], marker=mat0)    # 0

# geometry plot
cfv.figure(fig_size=(8,4))
cfv.title('Geometry')
cfv.draw_geometry(geometry)

""" Creating mesh from geometry object """
mesh = cfm.GmshMesh(geometry)

mesh.el_type = 2                            # type of element: 2 = triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.2

coords, edof, dofs, bdofs, elementmarkers = mesh.create()   # create the geometry
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)

# mesh plot
cfv.figure(fig_size=(8,4))
cfv.title('Mesh')
cfv.draw_mesh(coords=coords, edof=edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, filled=True)

""" Nodes indexing separated by boundary conditions """
bl = np.asarray(bdofs[left]) - 1                # index of nodes on left boundary
br = np.asarray(bdofs[right]) - 1               # index of nodes on right boundary
bb = np.asarray(bdofs[bottom]) - 1
bb = np.setdiff1d(bb, [0,1])
bt = np.asarray(bdofs[top]) - 1
bt = np.setdiff1d(bt, [2,3])

B = np.hstack((bl,br,bb,bt))                    # all boundaries

elementmarkers = np.asarray(elementmarkers)

bm0 = faces[elementmarkers == mat0]
bm0 = bm0.flatten()
bm0 = np.setdiff1d(bm0,B)

from plots import plot_nodes
plot_nodes(
    coords,
    b=(bl,br,bb,bt,bm0),
    labels=(
        "Left Dirichlet",
        "Right Dirichlet",
        "Bottom Neumann",
        "Top Neumann",
        "Material 0"
    ),
    figsize=(8,4),
    size=150,
    nums=True,
    alpha=0.75
)

""" Problem parameters """
# L = [A, B, C, 2D, E, 2F] is the coefitiens vector from GFDM that aproximates
# a differential lineal operator as:
# Au + Bu_x + Cu_y + Du_xx + Eu_xy + Fu_yy
L = np.array([0,0,0,2,0,2])
k0 = 1
k1 = 1
source = lambda p: -1
fl = lambda p: 1
fr = lambda p: 0
fn = lambda p: 0

materials = {}
materials["0"] = [k0, bm0]

neumann_boundaries = {}
neumann_boundaries["0"] = [k0, bt, fn]
neumann_boundaries["1"] = [k0, bb, fn]

dirichlet_boundaries = {}
dirichlet_boundaries["0"] = [bl, fl]
dirichlet_boundaries["1"] = [br, fr]


""" System `KU=F` assembling """
from GFDMI import create_system_K_F
K,F = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L,
    source=source,
    materials=materials,
    neumann_boundaries=neumann_boundaries,
    dirichlet_boundaries=dirichlet_boundaries
)

U = np.linalg.solve(K,F)

from plots import tri_surface
tri_surface(p=coords, t=faces, U=U, azim=-60, elev=30)

plt.show()