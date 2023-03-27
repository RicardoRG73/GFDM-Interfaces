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
g.point([0, 0])    # 0
g.point([0.5, 0], el_size=0.5)    # 1
g.point([2, 0])    # 2
g.point([2, 1])    # 3
g.point([1.5, 1], el_size=0.5)    # 4
g.point([0, 1])    # 5

# lines
left = 11       # marker for nodes on left boundary
right = 12      # marker for nodes on right boundary
neumann = 13    # marker for nodes on neumman boundaries (up, down)
interface = 14  # marker for nodes on interface
g.spline([0,1], marker=neumann)    # 0
g.spline([1,2], marker=neumann)    # 1
g.spline([2,3], marker=right)      # 2
g.spline([3,4], marker=neumann)    # 3
g.spline([4,5], marker=neumann)    # 4
g.spline([5,0], marker=left)       # 5
g.spline([1,4], marker=interface)  # 6

# surfaces
mat0 = 100      # marker for nodes on material 1
mat1 = 101      # marker for nodes on material 2
g.surface([0, 6, 4, 5], marker=mat0)    # 0
g.surface([1, 2, 3, 6], marker=mat1)    # 1

# geometry plot
cfv.figure(fig_size=(8,4))
cfv.title('Geometry')
cfv.draw_geometry(g)

plt.show()