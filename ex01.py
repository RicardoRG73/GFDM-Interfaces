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
delta_interface = 1e-1                      # separation beetween interface0 and interface1
delta_interface = delta_interface/2
geometry.point([-1,0])                      # 0
geometry.point([-0.5-delta_interface,0])    # 1
geometry.point([-0.5+delta_interface,0])    # 2
geometry.point([1,0])                       # 3
geometry.point([1,1])                       # 4
geometry.point([0.5+delta_interface,1])     # 5
geometry.point([0.5-delta_interface,1])     # 6
geometry.point([-1,1])                      # 7

# lines
left = 11                                   # marker for nodes on left boundary
right = 12                                  # marker for nodes on right boundary
neumann0 = 13
neumann1 = 14
neumann2 = 15
neumann3 = 16
interface0 = 17                             # marker for nodes on interface0
interface1 = 18                             # marker for nodes on interface1
geometry.spline([0,1], marker=neumann0)      # 0
geometry.spline([1,6], marker=interface0)   # 1
geometry.spline([6,7], marker=neumann1)      # 2
geometry.spline([7,0], marker=left)         # 3
geometry.spline([2,3], marker=neumann2)      # 4
geometry.spline([3,4], marker=right)        # 5
geometry.spline([4,5], marker=neumann3)      # 6
geometry.spline([2,5], marker=interface1)   # 7

# surfaces
mat0 = 100                                  # marker for nodes on material 1
mat1 = 101                                  # marker for nodes on material 2
geometry.surface([0,1,2,3], marker=mat0)    # 0
geometry.surface([4,5,6,7], marker=mat1)    # 1

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
bn0 = np.asarray(bdofs[neumann0]) - 1
bn0 = np.setdiff1d(bn0, [0,1])
bn1 = np.asarray(bdofs[neumann1]) - 1
bn1 = np.setdiff1d(bn1, [6,7])
bn2 = np.asarray(bdofs[neumann2]) - 1
bn2 = np.setdiff1d(bn2, [2,3])
bn3 = np.asarray(bdofs[neumann3]) - 1
bn3 = np.setdiff1d(bn3, [4,5])
bi0 = np.sort(np.asarray(bdofs[interface0])) - 1           # index of nodes on the interface in order
bi1 = np.sort(np.asarray(bdofs[interface1])) - 1

B = np.hstack((bl,br,bn0,bn1,bn2,bn3,bi0,bi1))                    # all boundaries

elementmarkers = np.asarray(elementmarkers)

m0 = faces[elementmarkers == mat0]
m0 = m0.flatten()
m0 = np.setdiff1d(m0,B)

m1 = faces[elementmarkers == mat1]
m1 = m1.flatten()
m1 = np.setdiff1d(m1,B)

from plots import plot_nodes
plot_nodes(
    coords,
    b=(bl,br,bn0,bn1,bn2,bn3,bi0,bi1,m0,m1),
    labels=(
        "Left",
        "Right",
        "Neumann 0",
        "Neumann 1",
        "Neumann 2",
        "Neumann 3",
        "Interface 0",
        "Interface 1",
        "Material 0",
        "Material 1"
    ),
    figsize=(8,4),
    size=150,
    nums=True,
    alpha=0.75,
)

""" Problem parameters """
# L = [A, B, C, 2D, E, 2F] is the coefitiens vector from GFDM that aproximates
# a differential lineal operator as:
# Au + Bu_x + Cu_y + Du_xx + Eu_xy + Fu_yy
L = np.array([0,0,0,2,0,2])
k0 = 1
k1 = 1
source = lambda p: 1
fl = lambda p: 1
fr = lambda p: 0
fn = lambda p: 0
beta = 0
fi0 = lambda p: beta
fi1 = lambda p: -beta

materials = {}
materials["0"] = [k0, m0]
materials["1"] = [k1, m1]

neumann_boundaries = {}
neumann_boundaries["0"] = [k0, bn0, fn]
neumann_boundaries["1"] = [k0, bn1, fn]
neumann_boundaries["2"] = [k1, bn2, fn]
neumann_boundaries["3"] = [k1, bn3, fn]

dirichlet_boundaries = {}
dirichlet_boundaries["0"] = [bl, fl]
dirichlet_boundaries["1"] = [br, fr]

interfaces = {}
interfaces["0"] = [k0, bi0, fi0, k1, bi1, fi1]


""" System `KU=F` assembling """
from GFDMI import create_system_K_F
K,F = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L,
    source=source,
    materials=materials,
    neumann_boundaries=neumann_boundaries,
    dirichlet_boundaries=dirichlet_boundaries,
    interfaces=interfaces
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
    z=U,
    i=faces[:,0],
    j=faces[:,1],
    k=faces[:,2],
)])
fig.show()

plt.show()