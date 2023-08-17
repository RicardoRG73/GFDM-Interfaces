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
ipf = 0.5                                   # interface point offset
geometry.point([-1,0])                      # 0
geometry.point([-ipf,0], el_size=0.2)                    # 1
geometry.point([1,0])                       # 2
geometry.point([1,1])                       # 3
geometry.point([ipf,1], el_size=0.2)                     # 4
geometry.point([-1,1])                      # 5

# lines
left = 11                                   # marker for nodes on left boundary
right = 12                                  # marker for nodes on right boundary
neumann0 = 13
neumann1 = 14
neumann2 = 15
neumann3 = 16
interface = 17                              # marker for nodes on interface0
geometry.spline([0,1], marker=neumann0)     # 0
geometry.spline([1,2], marker=neumann1)     # 1
geometry.spline([2,3], marker=right)        # 2
geometry.spline([3,4], marker=neumann2)     # 3
geometry.spline([4,5], marker=neumann3)     # 4
geometry.spline([5,0], marker=left)         # 5
geometry.spline([1,4], marker=interface)    # 6

# surfaces
mat0 = 100                                  # marker for nodes on material 1
mat1 = 101                                  # marker for nodes on material 2
geometry.surface([0,6,4,5], marker=mat0)    # 0
geometry.surface([1,2,3,6], marker=mat1)    # 1

# geometry plot
cfv.figure(fig_size=(8,4))
cfv.title('Geometry')
cfv.draw_geometry(geometry)

""" Creating mesh from geometry object """
mesh = cfm.GmshMesh(geometry)

mesh.el_type = 2                            # type of element: 2 = triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.1

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
bn1 = np.setdiff1d(bn1, [1,2])
bn2 = np.asarray(bdofs[neumann2]) - 1
bn2 = np.setdiff1d(bn2, [3,4])
bn3 = np.asarray(bdofs[neumann3]) - 1
bn3 = np.setdiff1d(bn3, [4,5])
bi = np.sort(np.asarray(bdofs[interface])) - 1           # index of nodes on the interface in order

B = np.hstack((bl,br,bn0,bn1,bn2,bn3,bi))                    # all boundaries

elementmarkers = np.asarray(elementmarkers)

m0 = faces[elementmarkers == mat0]
m0 = m0.flatten()
m0 = np.setdiff1d(m0,B)
m0 = np.hstack((m0,bn0,bn3))

m1 = faces[elementmarkers == mat1]
m1 = m1.flatten()
m1 = np.setdiff1d(m1,B)
m1 = np.hstack((m1,bn1,bn2))

from plots import plot_nodes
sizeA = 10
sizeB = 50
plot_nodes(
    coords,
    b=(bl,br,bn0,bn1,bn2,bn3,bi,m0,m1),
    labels=(
        "Left",
        "Right",
        "Neumann 0",
        "Neumann 1",
        "Neumann 2",
        "Neumann 3",
        "Interface",
        "Material 0",
        "Material 1"
    ),
    size=(sizeA,sizeA,sizeB,sizeB,sizeB,sizeB),
    nums=False
)

from plots import plot_normal_vectors
plot_normal_vectors(bi,coords)

""" Problem parameters """
# L = [A, B, C, 2D, E, 2F] is the coefitiens vector from GFDM that aproximates
# a differential lineal operator as:
# Au + Bu_x + Cu_y + Du_xx + Eu_xy + Fu_yy
L = np.array([0,0,0,1,0,1])
k0 = lambda p: 1
k1 = lambda p: 1
source = lambda p: 0
ul = lambda p: 1
ur = lambda p: 0
un = lambda p: 0
beta = lambda p: 0
alpha = lambda p: 0

materials = {}
materials["0"] = [k0, m0]
materials["1"] = [k1, m1]

neumann_boundaries = {}
neumann_boundaries["0"] = [k0, bn0, un]
neumann_boundaries["1"] = [k0, bn1, un]
neumann_boundaries["2"] = [k1, bn2, un]
neumann_boundaries["3"] = [k1, bn3, un]

dirichlet_boundaries = {}
dirichlet_boundaries["0"] = [bl, ul]
dirichlet_boundaries["1"] = [br, ur]

interfaces = {}
interfaces["0"] = [k0, k1, bi, beta, alpha, "0", "1"]   # = [difusion coefficient mat0, difusion coefficient mat1, index interface nodes, flux u_n jump, function u jump, material key 0, material key 1]


""" System `KU=F` assembling """
from GFDMI import create_system_K_F
K, F, U, p = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L,
    source=source,
    materials=materials,
    neumann_boundaries=neumann_boundaries,
    dirichlet_boundaries=dirichlet_boundaries,
    interfaces=interfaces
)

from plots import tri_surface
tri_surface(p=p, t=faces, U=U, azim=-60, elev=30)

from plots import contourf_plot
contourf_plot(p=p, U=U, levels=30)
plt.scatter(p[bi,0],p[bi,1], alpha=0.5, s=5)

plt.show()