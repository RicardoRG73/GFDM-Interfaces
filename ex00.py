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
geometry.point([0,0])                       # 1
geometry.point([1,0])                       # 2
geometry.point([0,1])                       # 3
geometry.point([-1,1])                      # 4

# lines
left = 11                                   # marker for nodes on left boundary
right = 12                                  # marker for nodes on right boundary
bottom = 13
top = 14
geometry.spline([0,1], marker=bottom)       # 0
geometry.spline([1,2], marker=bottom)       # 1
geometry.circle([2,1,3], marker=right)      # 2
geometry.spline([3,4], marker=top)          # 3
geometry.spline([4,0], marker=left)         # 4


# surfaces
mat0 = 100                                  # marker for nodes on material 1
geometry.surface([0,1,2,3,4], marker=mat0)  # 0

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
bb = np.setdiff1d(bb, [0,2])
bt = np.asarray(bdofs[top]) - 1
bt = np.setdiff1d(bt, [3,4])

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
# \mathb{L}u = Au + Bu_{x} + Cu_{y} + Du_{xx} + Eu_{xy} + Fu_{yy}
L = np.array([0,0,0,2,0,2])
k0 = 1
k1 = 1
source = lambda p: -5
fl = lambda p: p[0]
fr = lambda p: p[0]
fb = lambda p: p[0]
ft = lambda p: p[0]

materials = {}
materials["0"] = [k0, bm0]

neumann_boundaries = {}
#neumann_boundaries["0"] = [k0, bt, fn]
#neumann_boundaries["1"] = [k0, bb, fn]

dirichlet_boundaries = {}
dirichlet_boundaries["0"] = [bl, fl]
dirichlet_boundaries["1"] = [br, fr]
dirichlet_boundaries["top"] = [bt, ft]
dirichlet_boundaries["bottom"] = [bb, fb]


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

import plotly.graph_objects as go
fig = go.Figure(
    data=[
    go.Mesh3d(
    x=coords[:,0],
    y=coords[:,1],
    z=U
)])
fig.show()

plt.show()