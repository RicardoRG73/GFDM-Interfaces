"""
Laplace equation with sinusoidal interface, and jumps in `u` and `u_n`
[]

$ \nabla^2 u = f $

with source:

$ f(x,y) = -2\pi^2\sin(\pi x)\sin(\pi y) $

jump conditions:

$ u^+ - u^- = \sin(\pi x)e^{\pi y} $

$
\beta^+ \frac{\partial}{\partial n}u^+
-\beta^- \frac{\partial}{\partial n}u^-
=
\pi(
    \cos(\pi x)e^{\pi y}n_x
    + \sin(\pi x)e^{\pi y}n_y
)
$

where
$\beta^+ = \beta^- = 1$

The exact solution is given by

u(x,y) = 
\sin(\pi x)\sin(\pi y)  in  \Omega^+
\sin(\pi x)(\sin(\pi y) - e^{\pi y})  in  \Omega^-

The interface is crated by using
X = 0.5 + 0.1\sin(6.28 y)
Y = y
"""

# =============================================================================
# Importing nedeed libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

# calfem-python
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

# =============================================================================
# Creating geometry object
# =============================================================================
geometry = cfg.Geometry()

# points: square domain
geometry.point([0,0])    # 0
geometry.point([1,0])     # 1
geometry.point([1,1])      # 2
geometry.point([0,1])     # 3

# points: interface-boundaries intersection
delta_interface = 0.1
geometry.point([0.5 - delta_interface, 0])     # 4
geometry.point([0.5 - delta_interface, 1])     # 5
geometry.point([0.5 + delta_interface, 0])     # 6
geometry.point([0.5 + delta_interface, 1])     # 7

# lines: square domain
dirichlet = 10
geometry.spline([5,3], marker=dirichlet)    # 0
geometry.spline([3,0], marker=dirichlet)    # 1
geometry.spline([0,4], marker=dirichlet)    # 2

geometry.spline([6,1], marker=dirichlet)    # 3
geometry.spline([1,2], marker=dirichlet)    # 4
geometry.spline([2,7], marker=dirichlet)    # 5

# interface
## left interface
interface_left = 11
### points
N = 4
delta_y = 1/(N+1)
y = 0

for i in range(N):
    y += delta_y
    x = - delta_interface + 0.5 + 0.1 * np.sin(6.28 * y)
    geometry.point([x, y])

### lines
for i in range(N-1):
    geometry.spline([8+i,9+i], marker=interface_left)

### left interface lines, conecting interface and boundaries
geometry.spline([4,8], marker=interface_left)
geometry.spline([7+N,5], marker=interface_left)

## right interface
interface_right = 12
### points
y = 0
for i in range(N):
    y += delta_y
    x = delta_interface + 0.5 + 0.1 * np.sin(6.28 * y)
    geometry.point([x, y])

### lines
for i in range(N-1):
    geometry.spline([8+N+i,9+N+i], marker=interface_right)

### left interface lines, conecting interface and boundaries
geometry.spline([6,8+N], marker=interface_right)
geometry.spline([7+2*N,7], marker=interface_right)
    


# surfaces
## \Omega^+ : left side
left_domain = 0
left_surf_index = np.hstack((
    np.array([0,1,2]),
    np.array([5+N]),
    np.arange(5+1,5+N),
    np.array([5+N+1])
))
geometry.surface(left_surf_index, marker=left_domain)

## \Omega^- : right side
right_domain = 1
left_surf_index = np.hstack((
    np.array([5+2*N+1]),
    np.arange(5+N+2,5+2*N+1),
    np.array([5+2*N+2]),
    np.array([5,4,3])
))
geometry.surface(left_surf_index, marker=right_domain)

# geometry plot
cfv.figure(fig_size=(4,4))
cfv.title('Geometry')
cfv.draw_geometry(geometry)

# =============================================================================
# Creating mesh
# =============================================================================
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

# =============================================================================
# Nodes indexing separated by boundary conditions
# =============================================================================
# Dirichlet nodes
bd = np.asarray(bdofs[dirichlet]) - 1

# Interface nodes
bil = np.asarray(bdofs[interface_left]) - 1
bil = np.setdiff1d(bil, [4,5])
bir = np.asarray(bdofs[interface_right]) - 1
bir = np.setdiff1d(bir, [6,7])

# Interior nodes
elementmarkers = np.asarray(elementmarkers)
B = np.hstack((bd,bil,bir))

ml = faces[elementmarkers == left_domain]
ml = ml.flatten()
ml = np.setdiff1d(ml,B)

mr = faces[elementmarkers == right_domain]
mr = mr.flatten()
mr = np.setdiff1d(mr,B)

from plots import plot_nodes
plot_nodes(
    p=coords,
    b=(
       bd,
       bil,
       bir,
       ml,
       mr
    ),
    labels=(
        "Dirichlet",
        "Interface left",
        "Interface right",
        r"$\Omega^+$",
        r"$\Omega^-$"
    ),
    alpha = 0.3,
    loc="center",
    nums=True,
    title="Total nodes $N = "+ str(coords.shape[0])+"$"
)

from plots import plot_normal_vectors
plot_normal_vectors(bil, coords)
plot_normal_vectors(bir, coords)
plot_normal_vectors(bd, coords)

plt.show()