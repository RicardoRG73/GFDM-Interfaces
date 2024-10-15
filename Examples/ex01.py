#%% Importing needed libraries
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

import GFDMI

#%% Creating geometry
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
neumann_bottom_left = 101
left_interface = 102
neumann_top_left = 103
geometry.spline([3,0], marker=left)             # 0
geometry.spline([0,4], marker=neumann_bottom_left)         # 1
geometry.spline([4,6], marker=left_interface)       # 2
geometry.spline([6,3], marker=neumann_top_left)         # 3

right = 104
neumann_top_right = 105
right_interface = 106
neumann_bottom_right = 107
geometry.spline([1,2], marker=right)            # 4
geometry.spline([2,7], marker=neumann_top_right)         # 5
geometry.spline([5,7], marker=right_interface)       # 6
geometry.spline([5,1], marker=neumann_bottom_right)         # 7

# surfaces
mat0 = 10
mat1 = 11
geometry.surface([0,1,2,3], marker=mat0)        # 0
geometry.surface([4,5,6,7], marker=mat1)        # 1

#%% geometry plot
cfv.figure(fig_size=(8,4))
cfv.title('Geometry')
cfv.draw_geometry(geometry)
plt.savefig("figures/01geometry.jpg", dpi=300)

#%% Creating mesh
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

#%% mesh plot
cfv.figure(fig_size=(8,4))
cfv.title('Mesh')
cfv.draw_mesh(coords=coords, edof=edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, filled=True)
plt.savefig("figures/01mesh.jpg", dpi=300)

#%% Nodes indexing separated by boundary conditions
# Dirichlet nodes
left_nodes = np.asarray(bdofs[left]) - 1                # index of nodes on left boundary
right_nodes = np.asarray(bdofs[right]) - 1               # index of nodes on right boundary

# Neumann nodes
bottom_left_nodes = np.asarray(bdofs[neumann_bottom_left]) - 1
bottom_left_nodes = np.setdiff1d(bottom_left_nodes, 0)
top_left_nodes = np.asarray(bdofs[neumann_top_left]) - 1
top_left_nodes = np.setdiff1d(top_left_nodes, 3)
top_right_nodes = np.asarray(bdofs[neumann_top_right]) - 1
top_right_nodes = np.setdiff1d(top_right_nodes, 2)
bottom_right_nodes = np.asarray(bdofs[neumann_bottom_right]) - 1
bottom_right_nodes = np.setdiff1d(bottom_right_nodes, 1)

# Interface nodes
left_interface_nodes = np.asarray(bdofs[left_interface]) - 1
left_interface_nodes = np.setdiff1d(left_interface_nodes, [4,6])
right_interface_nodes = np.asarray(bdofs[right_interface]) - 1
right_interface_nodes = np.setdiff1d(right_interface_nodes, [5,7])

# Interior nodes
elementmarkers = np.asarray(elementmarkers)
boundaries = np.hstack((
    left_nodes,
    right_nodes,
    bottom_left_nodes,
    top_left_nodes,
    top_right_nodes,
    bottom_right_nodes,
    left_interface_nodes,
    right_interface_nodes
))

interior_nodes_mat0 = faces[elementmarkers == mat0]
interior_nodes_mat0 = interior_nodes_mat0.flatten()
interior_nodes_mat0 = np.setdiff1d(interior_nodes_mat0,boundaries)

interior_nodes_mat1 = faces[elementmarkers == mat1]
interior_nodes_mat1 = interior_nodes_mat1.flatten()
interior_nodes_mat1 = np.setdiff1d(interior_nodes_mat1,boundaries)

#%% ploting boundaries in different colors
plt.figure()
nodes = (
    left_nodes,
    right_nodes,
    bottom_left_nodes,
    top_left_nodes,
    top_right_nodes,
    bottom_right_nodes,
    left_interface_nodes,
    right_interface_nodes,
    interior_nodes_mat0,
    interior_nodes_mat1
)
labels = (
    "Left",
    "Right",
    "Bottom-Left",
    "Top-Left",
    "Top-Right",
    "Bottom-Right",
    "Left Interface",
    "Right Interface",
    "Interior Material 0",
    "Interior Material 1"
)
for b,label in zip(nodes, labels):
    plt.scatter(coords[b,0], coords[b,1], label=label, alpha=0.5, s=20)
plt.axis("equal")
plt.title("$N = %d$" %coords.shape[0])
plt.legend(loc="center")
plt.savefig("figures/01nodes.jpg", dpi=300)

#%% plotting normal vectors at interface
plt.figure()
nodes_to_plot = (left_interface_nodes, right_interface_nodes)
labels = ("Left Interface", "Right Interface")
colors = ("#394CC9", "#C92520")
for nodes,label,color in zip(nodes_to_plot,labels,colors):
    norm_vec = GFDMI.normal_vectors(nodes,coords)
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
plt.savefig("figures/01normal.jpg", dpi=300)

#%% Problem parameters
# L = [A, B, C, 2D, E, 2F] is the coefitiens vector from GFDM that aproximates
# a differential lineal operator as:
# \mathb{L}u = Au + Bu_{x} + Cu_{y} + Du_{xx} + Eu_{xy} + Fu_{yy}
L = np.array([0,0,0,1,0,1])
permeability_mat0 = lambda p: 1
permeability_mat1 = lambda p: 0.1
source = lambda p: -1
left_condition = lambda p: 1 - p[1]**2
right_condition = lambda p: 1
bottom_condition = lambda p: 0
top_condition = lambda p: 0
# flux difference at interface du/dn|_{mat0} - du/dn|_{mat1} = beta
beta = lambda p: 0
# solution diference at interface u_{mat0} - u_{mat1} = alpha
alpha = lambda p: 0.5

#%% assembling boundary conditions in dictionaries
materials = {}
materials['material0'] = [permeability_mat0, interior_nodes_mat0]
materials['material1'] = [permeability_mat1, interior_nodes_mat1]

neumann_boundaries = {}
neumann_boundaries['bottom_left'] =   [permeability_mat0,   bottom_left_nodes,  bottom_condition]
neumann_boundaries['top_left'] =      [permeability_mat0,   top_left_nodes,     top_condition]
neumann_boundaries['top_right_'] =    [permeability_mat1,   top_right_nodes,    top_condition]
neumann_boundaries['bottom_right'] =  [permeability_mat1,   bottom_right_nodes, bottom_condition]

dirichlet_boundaries = {}
dirichlet_boundaries["left"] =  [left_nodes,    left_condition]
dirichlet_boundaries["right"] = [right_nodes,   right_condition]

interfaces = {}
interfaces["interface"] = [
    permeability_mat0,
    permeability_mat1,
    left_interface_nodes,
    right_interface_nodes,
    beta,
    alpha,
    interior_nodes_mat0,
    interior_nodes_mat1
]


#%% System `KU=F` assembling
K,F = GFDMI.create_system_K_F(
    p=coords,
    triangles=faces,
    L=L,
    source=source,
    materials=materials,
    neumann_boundaries=neumann_boundaries,
    dirichlet_boundaries=dirichlet_boundaries,
    interfaces = interfaces
)

#%% Solution
U = sp.linalg.spsolve(K,F)

#%% contourf plot
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
plt.savefig("figures/01contourf.jpg", dpi=300)

#%% 3d plot
fig = plt.figure()
ax = plt.axes(projection="3d")
surface = ax.plot_trisurf(
    coords[:,0],
    coords[:,1],
    U,
    cmap="plasma",
    aa=False
)
fig.colorbar(surface)
ax.view_init(30,-120)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("U")
plt.savefig("figures/01-3d.jpg", dpi=300)

#%%
plt.show()