#%% Importing Needed libraries
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

# from files
import GFDMI

#%% Creating geometry object
geometry = cfg.Geometry()                   # geometry object

# points
geometry.point([0,0])                      # 0
geometry.point([1,0])                       # 1
geometry.point([2,0])                       # 2
geometry.point([1,1])                       # 3
geometry.point([0,1])                      # 4

# lines
left = 11                                   # marker for nodes on left boundary
right = 12                                  # marker for nodes on right boundary
bottom = 13                                 # marker for bottom nodes
top = 14                                    # marker for top nodes
geometry.spline([0,1], marker=bottom)       # 0
geometry.spline([1,2], marker=bottom)       # 1
geometry.circle([2,1,3], marker=right)      # 2
geometry.spline([3,4], marker=top)          # 3
geometry.spline([4,0], marker=left)         # 4


# surfaces
mat0 = 100                                  # marker for nodes on material 1
geometry.surface([0,1,2,3,4], marker=mat0)  # 0

# geometry plot
cfv.figure()
cfv.title('Geometry')
cfv.draw_geometry(geometry)
# plt.savefig("figures/00geometry.jpg", dpi=300)

#%% Creating mesh from geometry object
mesh = cfm.GmshMesh(geometry)

mesh.el_type = 2                            # type of element: 2 = triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.08

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
# plt.savefig("figures/00mesh.jpg", dpi=300)

#%% Nodes indexing separated by boundary conditions
left_nodes = np.asarray(bdofs[left]) - 1                # index of nodes on left boundary
right_nodes = np.asarray(bdofs[right]) - 1               # index of nodes on right boundary
right_nodes = np.setdiff1d(right_nodes, [2,3])
bottom_nodes = np.asarray(bdofs[bottom]) - 1
bottom_nodes = np.setdiff1d(bottom_nodes, [0])
top_nodes = np.asarray(bdofs[top]) - 1
top_nodes = np.setdiff1d(top_nodes, [4])

B = np.hstack((
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_nodes
))                          # all boundaries

elementmarkers = np.asarray(elementmarkers)

interior_nodes = faces[elementmarkers == mat0]
interior_nodes = interior_nodes.flatten()
interior_nodes = np.setdiff1d(interior_nodes,B)

#%% ploting boundaries in different color
plt.figure()
nodes = (left_nodes,right_nodes,bottom_nodes,top_nodes,interior_nodes)
labels = ("Left", "Right", "Bottom", "Top", "Interior")
for b,label in zip(nodes, labels):
    plt.scatter(coords[b,0], coords[b,1], label=label)
plt.axis("equal")
plt.title("$N = %d$" %coords.shape[0])
plt.legend(loc="center")
# plt.savefig("figures/00nodes.jpg", dpi=300)

#%% ploting normal vectors
plt.figure()
norm_vec = GFDMI.normal_vectors(right_nodes,coords)
plt.scatter(coords[right_nodes,0], coords[right_nodes,1])
plt.quiver(
    coords[right_nodes,0],
    coords[right_nodes,1],
    norm_vec[:,0],
    norm_vec[:,1],
    alpha=0.75,
    color="#394C89"
)
plt.axis("equal")
# plt.savefig("figures/00normal.jpg", dpi=300)

#%% Problem parameters
# L = [A, B, C, 2D, E, 2F] is the coefitiens vector from GFDM that aproximates
# a differential lineal operator as:
# \mathb{L}u = Au + Bu_{x} + Cu_{y} + Du_{xx} + Eu_{xy} + Fu_{yy}
L = np.array([0,0,0,1,0,1])
permeability = lambda p: 1
source = lambda p: -2
left_condition = lambda p: 0
right_condition = lambda p: 0
bottom_condition = lambda p: p[0] * 0.5
top_condition = lambda p: p[0]

# assembling conditions into dictionaries
materials = {}
materials["0"] = [permeability, interior_nodes]

neumann_boundaries = {}
neumann_boundaries["right"] = [permeability, right_nodes, right_condition]

dirichlet_boundaries = {}
dirichlet_boundaries["0"] = [left_nodes, left_condition]
dirichlet_boundaries["top"] = [top_nodes, top_condition]
dirichlet_boundaries["bottom"] = [bottom_nodes, bottom_condition]


#%% System `KU=F` assembling
K,F = GFDMI.create_system_K_F(
    p=coords,
    triangles=faces,
    L=L,
    source=source,
    materials=materials,
    neumann_boundaries=neumann_boundaries,
    dirichlet_boundaries=dirichlet_boundaries
)

#%% Solution to KU=F
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
# plt.savefig("figures/00contourf.jpg", dpi=300)

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
ax.view_init(30,-130)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("U")
# plt.savefig("figures/00-3d.jpg", dpi=300)


#%%
plt.show()