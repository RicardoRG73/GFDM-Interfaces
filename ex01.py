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
neumann = 13                                # marker for nodes on neumman boundaries (up, down)
interface0 = 14                             # marker for nodes on interface0
interface1 = 15                             # marker for nodes on interface1
geometry.spline([0,1], marker=neumann)      # 0
geometry.spline([1,6], marker=interface0)   # 1
geometry.spline([6,7], marker=neumann)      # 2
geometry.spline([7,0], marker=left)         # 3
geometry.spline([2,3], marker=neumann)      # 4
geometry.spline([3,4], marker=right)        # 5
geometry.spline([4,5], marker=neumann)      # 6
geometry.spline([5,2], marker=interface1)   # 7

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
mesh.el_size_factor = 0.1

coords, edof, dofs, bdofs, elementmarkers = mesh.create()   # create the geometry
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)

# mesh plot
cfv.figure(fig_size=(7,7))
cfv.title('Mesh')
cfv.draw_mesh(coords=coords, edof=edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, filled=True)

""" Nodes indexing separated by boundary conditions """
bl = np.asarray(bdofs[left]) - 1                # index of nodes on left boundary
br = np.asarray(bdofs[right]) - 1               # index of nodes on right boundary
bn = np.asarray(bdofs[neumann]) - 1             # index of nodes on neumann boundaries
bn = np.setdiff1d(bn, np.arange(8))                # deletes nodes that are on the left boundary, right boundary, and interface
bi0 = np.sort(np.asarray(bdofs[interface0])) - 1           # index of nodes on the interface in order
bi1 = np.sort(np.asarray(bdofs[interface1])) - 1

B = np.hstack((bl,br,bn,bi0,bi1))                    # all boundaries

elementmarkers = np.asarray(elementmarkers)

bm0 = faces[elementmarkers == mat0]
bm0 = bm0.flatten()
bm0 = np.setdiff1d(bm0,B)

bm1 = faces[elementmarkers == mat1]
bm1 = bm1.flatten()
bm1 = np.setdiff1d(bm1,B)

from plots import plot_nodes
plot_nodes(
    coords,
    (bl, br, bn, bi0, bi1, bm0, bm1),
    ("Left", "Right", "Neumann", "Interface 0", "Interface 1", "Material 0", "Material 1"),
    figsize=(8,4),
    size=100,
    nums=False,
    alpha=0.5
)



plt.show()