""" Importing needed libraries """
import numpy as np
import matplotlib.pyplot as plt

# calfem-python
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

""" Creating geometry object """
g = cfg.Geometry()          # geometry object

# points
delta_i = 1e-1
g.point([-1,0])             # 0
g.point([-0.5-delta_i,0])   # 1
g.point([-0.5+delta_i,0])   # 2
g.point([1,0])              # 3
g.point([1,1])              # 4
g.point([0.5+delta_i,1])    # 5
g.point([0.5-delta_i,1])    # 6
g.point([-1,1])             # 7

# lines
left = 11                   # marker for nodes on left boundary
right = 12                  # marker for nodes on right boundary
neumann = 13                # marker for nodes on neumman boundaries (up, down)
interface = 14              # marker for nodes on interface
g.spline([0,1])             # 0
g.spline([1,6])             # 1
g.spline([6,7])             # 2
g.spline([7,0])             # 3
g.spline([2,3])             # 4
g.spline([3,4])             # 5
g.spline([4,5])             # 6
g.spline([5,2])             # 7

# surfaces
mat0 = 100                  # marker for nodes on material 1
mat1 = 101                  # marker for nodes on material 2
g.surface([0,1,2,3], marker=mat0)
g.surface([4,5,6,7], marker=mat1)

# geometry plot
cfv.figure(fig_size=(8,4))
cfv.title('Geometry')
cfv.draw_geometry(g)

mesh = cfm.GmshMesh(g)

mesh.el_type = 2                # type of element: 2 = triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.08

coords, edof, dofs, bdofs, elementmarkers = mesh.create()       # create the geometry
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(coords, edof, mesh.dofs_per_node, mesh.el_type)  # coordinate-edges to vertices-faces

# mesh plot
cfv.figure(fig_size=(7,7))
cfv.title('Mesh')
cfv.draw_mesh(coords=coords, edof=edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, filled=True)

plt.show()