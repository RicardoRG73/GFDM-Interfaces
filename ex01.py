""" Importing needed libraries """
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

# calfem-python
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv


""" Creating geometry """
geometry = cfg.Geometry()                       # geometry object

# points
geometry.point([-1,0])                          # 0
geometry.point([1,0])                           # 1
geometry.point([1,1])                           # 2
geometry.point([-1,1])                          # 3

delta = 0.0005   
interface_offset = 0.2
el_size_scale_factor = 1
geometry.point([-interface_offset-delta, 0], el_size=el_size_scale_factor)    # 4
geometry.point([-interface_offset+delta, 0], el_size=el_size_scale_factor)    # 5
geometry.point([interface_offset-delta, 1], el_size=el_size_scale_factor)     # 6
geometry.point([interface_offset+delta, 1], el_size=el_size_scale_factor)     # 7

# lines
left = 100
neumann0 = 101
interfaceA = 102
neumann1 = 103
geometry.spline([3,0], marker=left)             # 0
geometry.spline([0,4], marker=neumann0)         # 1
geometry.spline([4,6], marker=interfaceA)       # 2
geometry.spline([6,3], marker=neumann1)         # 3

right = 104
neumann2 = 105
interfaceB = 106
neumann3 = 107
geometry.spline([1,2], marker=right)            # 4
geometry.spline([2,7], marker=neumann2)         # 5
geometry.spline([5,7], marker=interfaceB)       # 6
geometry.spline([5,1], marker=neumann3)         # 7

# surfaces
mat0 = 10
mat1 = 11
geometry.surface([0,1,2,3], marker=mat0)        # 0
geometry.surface([4,5,6,7], marker=mat1)        # 1

# geometry plot
cfv.figure(fig_size=(8,4))
cfv.title('Geometry')
cfv.draw_geometry(geometry)


""" Creating mesh """
mesh = cfm.GmshMesh(geometry)

mesh.el_type = 2                            # type of element: 2 = triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.04

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
# Dirichlet nodes
bl = np.asarray(bdofs[left]) - 1                # index of nodes on left boundary
br = np.asarray(bdofs[right]) - 1               # index of nodes on right boundary

# Neumann nodes
bn0 = np.asarray(bdofs[neumann0]) - 1
bn0 = np.setdiff1d(bn0, np.array([0,4]))
bn1 = np.asarray(bdofs[neumann1]) - 1
bn1 = np.setdiff1d(bn1, np.array([6,3]))
bn2 = np.asarray(bdofs[neumann2]) - 1
bn2 = np.setdiff1d(bn2, np.array([2,7]))
bn3 = np.asarray(bdofs[neumann3]) - 1
bn3 = np.setdiff1d(bn3, np.array([5,1]))

# Interface nodes
biA = np.asarray(bdofs[interfaceA]) - 1
biB = np.asarray(bdofs[interfaceB]) - 1

# Interior nodes
elementmarkers = np.asarray(elementmarkers)
B = np.hstack((bl,br,bn0,bn1,bn2,bn3,biA,biB))

m0 = faces[elementmarkers == mat0]
m0 = m0.flatten()
m0 = np.setdiff1d(m0,B)

m1 = faces[elementmarkers == mat1]
m1 = m1.flatten()
m1 = np.setdiff1d(m1,B)

from plots import plot_nodes
plot_nodes(
    p=coords,
    b=(
       bl,
       br,
       bn0,
       bn1,
       bn2,
       bn3,
       biA,
       biB,
       m0,
       m1
    ),
    labels=(
        "Left",
        "Right",
        "Neu0",
        "Neu1",
        "Neu2",
        "Neu3",
        "Interface A",
        "Interface B",
        "Mat0",
        "Mat1"
    ),
    alpha = 0.5,
    loc="center",
    nums=True,
    title="Total nodes $N = "+ str(coords.shape[0])+"$"
)

from plots import plot_normal_vectors
plot_normal_vectors(biA, coords)
plot_normal_vectors(biB, coords)


""" Problem parameters """
# L = [A, B, C, 2D, E, 2F] is the coefitiens vector from GFDM that aproximates
# a differential lineal operator as:
# \mathb{L}u = Au + Bu_{x} + Cu_{y} + Du_{xx} + Eu_{xy} + Fu_{yy}
L = np.array([0,0,0,1,0,1])
k0 = lambda p: 1
k1 = lambda p: 1
source = lambda p: 0
fl = lambda p: 1
fr = lambda p: 0
fb = lambda p: 0
ft = lambda p: 0
beta = lambda p: 0.7
alpha = lambda p: -0.3

materials = {}
materials['material0'] = [k0, m0]
materials['material1'] = [k1, m1]

neumann_boundaries = {}
neumann_boundaries['neumann0'] = [k0, bn0, fb]
neumann_boundaries['neumann1'] = [k0, bn1, ft]
neumann_boundaries['neumann2'] = [k1, bn2, ft]
neumann_boundaries['neumann3'] = [k1, bn3, fb]

dirichlet_boundaries = {}
dirichlet_boundaries["left"] = [bl, fl]
dirichlet_boundaries["right"] = [br, fr]

interfaces = {}
interfaces["interface0"] = [k0, k1, biA, biB, beta, alpha, m0, m1]


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
    interfaces = interfaces
)

U = sp.linalg.spsolve(K,F)

from plots import tri_surface
tri_surface(p=coords, t=faces, U=U, azim=-60, elev=30, title="Solution using $N = "+ str(coords.shape[0])+"$")

from plots import contourf_plot
contourf_plot(p=coords, U=U, levels=30, title="Solution using $N = "+ str(coords.shape[0])+"$")
plt.scatter(coords[biA,0], coords[biA,1], alpha=0.45, s=5, color="black")
plt.scatter(coords[biB,0], coords[biB,1], alpha=0.45, s=5, color="white")

plt.show()