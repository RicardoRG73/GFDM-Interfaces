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
g.point([-1,0])
g.point([-0.5-delta_i,0])
g.point([-0.5+delta_i,0])
g.point([1,0])
g.point([1,1])
g.point([0.5+delta_i,1])
g.point([0.5-delta_i,1])
g.point([-1,1])

# lines
left = 11       # marker for nodes on left boundary
right = 12      # marker for nodes on right boundary
neumann = 13    # marker for nodes on neumman boundaries (up, down)
interface = 14  # marker for nodes on interface


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