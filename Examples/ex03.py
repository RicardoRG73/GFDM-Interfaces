"""
Laplace equation with sinusoidal interface, and jumps in `u` and `u_n`
[Siraj-ul-Islam, Masood Ahmad]

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

#%%
# =============================================================================
# Importing nedeed libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["figure.autolayout"] = True
import scipy.sparse as sp

# calfem-python
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

import GFDMIex03

#%%
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
delta_interface = 0.01

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
N = 11
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

#%% geometry plot
cfv.figure(fig_size=(4,4))
cfv.title('Geometry')
cfv.draw_geometry(geometry)

#%%
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

#%% mesh plot
cfv.figure(fig_size=(8,4))
cfv.title('Mesh')
cfv.draw_mesh(coords=coords, edof=edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, filled=True)

#%%
# =============================================================================
# Nodes indexing separated by boundary conditions
# =============================================================================
# Dirichlet nodes
dirichlet_nodes = np.asarray(bdofs[dirichlet]) - 1

# Interface nodes
left_interface_nodes = np.asarray(bdofs[interface_left]) - 1
left_interface_nodes = np.setdiff1d(left_interface_nodes, [4,5])
right_interface_nodes = np.asarray(bdofs[interface_right]) - 1
right_interface_nodes = np.setdiff1d(right_interface_nodes, [6,7])

# Interior nodes
elementmarkers = np.asarray(elementmarkers)
B = np.hstack((dirichlet_nodes,left_interface_nodes,right_interface_nodes))

left_interior_nodes = faces[elementmarkers == left_domain]
left_interior_nodes = left_interior_nodes.flatten()
left_interior_nodes = np.setdiff1d(left_interior_nodes,B)

right_interior_nodes = faces[elementmarkers == right_domain]
right_interior_nodes = right_interior_nodes.flatten()
right_interior_nodes = np.setdiff1d(right_interior_nodes,B)

#%% ploting boundaries in different colors
plt.figure()
nodes = (
    dirichlet_nodes,
    left_interface_nodes,
    right_interface_nodes,
    left_interior_nodes,
    right_interior_nodes
)
labels=(
    "Dirichlet",
    "Interface left",
    "Interface right",
    r"$\Omega^+$",
    r"$\Omega^-$"
)
for b,label in zip(nodes, labels):
    plt.scatter(coords[b,0], coords[b,1], label=label)
plt.axis("equal")
plt.title("$N = %d$" %coords.shape[0])
plt.legend()

#%% plotting normal vectors
plt.figure()

nodes_to_plot = (left_interface_nodes, right_interface_nodes)
labels = ("Left Interface", "Right Interface")
colors = ("#394CC9", "#C92520")
for nodes,label,color in zip(nodes_to_plot,labels,colors):
    norm_vec = GFDMIex03.normal_vectors(nodes,coords)
    plt.scatter(
        coords[nodes,0],
        coords[nodes,1],
        alpha=0.5,
        color=color,
        label=label
    )
    plt.quiver(
        coords[nodes,0],
        coords[nodes,1],
        norm_vec[:,0],
        norm_vec[:,1],
        alpha=0.5,
        color=color,
        label="Normal Vectors - " + label
    )
plt.axis("equal")
plt.legend()


#%% Problem parameters
# L = [A, B, C, 2D, E, 2F] is the coefitiens vector from GFDM that aproximates
# a differential lineal operator as:
# \mathb{L}u = Au + Bu_{x} + Cu_{y} + Du_{xx} + Eu_{xy} + Fu_{yy}
L = np.array([0,0,0,1,0,1])
permeability_left = lambda p: 1
permeability_right = lambda p: 1
source = lambda p: 4

def dirichlet_condition(p):
    if p[0] < 0.5:
        value = np.sin(np.pi*p[0]) * np.sin(np.pi*p[1])
    else:
        value = np.sin(np.pi*p[0]) * (
            np.sin(np.pi*p[1])
            - np.exp(np.pi*p[1])
        )
    return value

# flux diference du/dn|_{left} - du/dn|_{rignt} = beta 
def beta(p):
    n = GFDMIex03.normal_vectors(left_interface_nodes, coords)
    i = np.argmin(
        np.sqrt(
            (coords[left_interface_nodes,0]-p[0])**2
            +
            (coords[left_interface_nodes,1]-p[1])**2
        )
    )
    value = np.pi * (
        np.cos(np.pi*p[0])
        * np.exp(np.pi*p[1])
        * n[i,0]
        + np.sin(np.pi*p[0])
        * np.exp(np.pi*p[1])
        * n[i,1]
    )
    return value

# solution difference u|_{left} - u|_{right} = alpha
alpha = lambda p: -np.sin(np.pi*p[0]) * np.exp(np.pi*p[1])

#%% assembling boundary conditions in dictionaries
materials = {}
materials['material_left'] = [permeability_left, left_interior_nodes]
materials['material_right'] = [permeability_right, right_interior_nodes]

neumann_boundaries = {}

dirichlet_boundaries = {}
dirichlet_boundaries["dirichlet"] = [dirichlet_nodes, dirichlet_condition]

interfaces = {}
interfaces["interface0"] = [permeability_left, permeability_right, left_interface_nodes, right_interface_nodes, beta, alpha, left_interior_nodes, right_interior_nodes]


#%% Assembling system `KU=F`
from GFDMIex03 import create_system_K_F
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

#%% Solution of system `KU=F`
U = sp.linalg.spsolve(K,F)

#%% contourf
fig = plt.figure()
ax = plt.axes()
cont = ax.tricontourf(
    coords[:,0],
    coords[:,1],
    U,
    cmap="plasma",
    levels=11
)
fig.colorbar(cont)
cont = ax.tricontour(
    coords[:,0],
    coords[:,1],
    U,
    colors="k",
    levels=11
)
plt.clabel(cont, inline=True)
plt.axis("equal")
plt.xlabel("x")
plt.ylabel("y")

#%% exact solution
def exact(p):
    if p[0] <= 0.5 + 0.1 * np.sin(6.28 * p[1]):
        value = np.sin(np.pi*p[0]) * np.sin(np.pi*p[1])
    else:
        value = np.sin(np.pi*p[0]) * (
            np.sin(np.pi*p[1])
            - np.exp(np.pi*p[1])
        )
    return value

Uex = np.zeros(shape=U.shape)
for i in range(U.shape[0]):
    Uex[i] = exact(coords[i,:])

#%% 3D plotting
fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_trisurf(
    coords[:,0],
    coords[:,1],
    U,
    color="r",
    alpha=0.5,
    aa=False,
    label="Numerical"
)
ax.plot_trisurf(
    coords[:,0],
    coords[:,1],
    Uex,
    color="b",
    alpha=0.5,
    aa=False,
    label="Exact"
)
plt.legend()
ax.view_init(20,-50)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("U")


#%% Root Mean Square Error
RMSE = np.sqrt(
    np.mean(
        (Uex - U)**2
    )
)

print("N = %d" %coords.shape[0])
print("\n===============")
print("RMSE = %1.4e" %RMSE)
print("===============")

# Norm 2
n2 = np.linalg.norm(Uex-U)
print("\n===============")
print("Norm 2 = %1.4e" %n2)
print("===============")

# Norm infty
ninf = np.max(np.max(np.abs(Uex-U)))
print("\n===============")
print("Norm 2 = %1.4e" %ninf)
print("===============")

#%%
plt.show()