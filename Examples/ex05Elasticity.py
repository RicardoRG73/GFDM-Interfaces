#%%
# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

import GFDMI

#%%
# =============================================================================
# Geometry
# =============================================================================
g = cfg.Geometry()

# points
g.point([0,0])      # 0
g.point([2,0])      # 1
g.point([2,0.4])      # 2
g.point([0,0.4])      # 3

# lines
left = 10
right = 11
top = 12
bottom = 13

g.line([0,1], marker=bottom)    # 0
g.line([1,2], marker=right)     # 1
g.line([2,3], marker=top)       # 2
g.line([3,0], marker=left)      # 3


# surfaces
mat0 = 0
g.surface([0,1,2,3], marker=mat0)


#%% plotting geometry
cfv.figure()
cfv.title('Geometry')
cfv.draw_geometry(g, draw_axis=True)
plt.savefig("figures/05geometry.jpg", dpi=300)

#%%
# =============================================================================
# Mesh
# =============================================================================
mesh = cfm.GmshMesh(g,el_size_factor=0.1)

coords, edof, dofs, bdofs, elementmarkers = mesh.create()
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)

#%% plotting mesh
cfv.figure()
cfv.title('Mesh $N=%d' %coords.shape[0] +'$')
cfv.draw_mesh(
    coords=coords,
    edof=edof,
    dofs_per_node=mesh.dofs_per_node,
    el_type=mesh.el_type,
    filled=True
)
plt.savefig("figures/05mesh.jpg", dpi=300)


#%%
# =============================================================================
# Nodes identification by color
# =============================================================================
corner_nodes = np.array([0,1,2,3])

left_nodes = np.asarray(bdofs[left]) - 1
left_nodes = np.setdiff1d(left_nodes, corner_nodes)
left_nodes = np.hstack((left_nodes, [0,3]))

right_nodes = np.asarray(bdofs[right]) - 1
right_nodes = np.setdiff1d(right_nodes, corner_nodes)
right_nodes = np.hstack((right_nodes, [1,2]))

bottom_nodes = np.asarray(bdofs[bottom]) - 1
bottom_nodes = np.setdiff1d(bottom_nodes, corner_nodes)

top_nodes = np.asarray(bdofs[top]) - 1
top_nodes = np.setdiff1d(top_nodes, corner_nodes)

boundaries = np.hstack((
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_nodes
))

N = coords.shape[0]
interior_nodes = np.setdiff1d(np.arange(N), boundaries)

#%% plotting nodes by color
plt.figure()
nodes_to_plot = (
    interior_nodes,
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_nodes
)
labels = (
    "Interior",
    "Left",
    "Right",
    "Bottom",
    "Top"
)
for nodes,label in zip(nodes_to_plot, labels):
    plt.scatter(
        coords[nodes,0],
        coords[nodes,1],
        label=label,
        alpha=0.75,
        s=10
)
plt.axis("equal")
plt.legend()
plt.savefig("figures/05nodes.jpg", dpi=300)

#%%
# =============================================================================
# Problem parameters
# u: horizontal displacement
# v: vertical displacement
# =============================================================================
k = lambda p: 1
source_u = lambda p: 0
source_v = lambda p: 0#-3e-8

# E: young modulus, nu: poisson constant
E = 22e9
nu = 0.2

# PS: plain-strain matrix
alpha = E * (1 - nu) / (1 + nu) / (1 - 2*nu)
PS = np.array([
    [1, nu/(1-nu), 0],
    [nu/(1-nu), 1, 0],
    [0, 0, (1-2*nu)/2/(1-nu)]
])
PS *= alpha

# coefitiens vectors L
L11 = [0,0,0,PS[0,0],PS[0,2]+PS[2,0],PS[2,2]]
L12 = [0,0,0,PS[0,2],PS[0,1]+PS[2,2],PS[2,1]]
L21 = [0,0,0,PS[2,0],PS[1,0]+PS[2,2],PS[1,2]]
L22 = [0,0,0,PS[2,2],PS[1,2]+PS[2,1],PS[1,1]]

#%%
# =============================================================================
# Boundary conditions
# =============================================================================
dirichlet_condition = lambda p: 0
neumann_condition = lambda p: 0
neumann_load = lambda p: 5e-15

#%%
# =============================================================================
# Discretization with GFDM
# =============================================================================
# boundary condition assembled as dictionaries
materials = {}
materials["0"] = [k, interior_nodes]

u_Dir = {}
u_Dir["left"] = [left_nodes, dirichlet_condition]
# uDir["right"] = [nodesr, fDir]

u_Neu = {}
u_Neu["bottom"] = [k, bottom_nodes, neumann_condition]
u_Neu["right"] = [k, right_nodes, neumann_condition]
u_Neu["top"] = [k, top_nodes, neumann_condition]

v_Dir = {}
v_Dir["left"] = [left_nodes, dirichlet_condition]
# vDir["right"] = [nodesr, fDir]

v_Neu = {}
v_Neu["bottom"] = [k, bottom_nodes, neumann_condition]
v_Neu["right"] = [k, right_nodes, neumann_condition]
v_Neu["top"] = [k, top_nodes, neumann_load]

#%% sistem KU=F assembling
D11, F11 = GFDMI.create_system_K_F(
    p=coords,
    triangles=faces,
    L=L11,
    source=source_u,
    materials=materials,
    dirichlet_boundaries=u_Dir,
    neumann_boundaries=u_Neu
)
D12, F12 = GFDMI.create_system_K_F(
    p=coords,
    triangles=faces,
    L=L12,
    source=source_v,
    materials=materials,
    dirichlet_boundaries=v_Dir,
    neumann_boundaries=v_Neu
)
D21, F21 = GFDMI.create_system_K_F(
    p=coords,
    triangles=faces,
    L=L21,
    source=source_u,
    materials=materials,
    dirichlet_boundaries=u_Dir,
    neumann_boundaries=u_Neu
)
D22, F22 = GFDMI.create_system_K_F(
    p=coords,
    triangles=faces,
    L=L12,
    source=source_v,
    materials=materials,
    dirichlet_boundaries=v_Dir,
    neumann_boundaries=v_Neu
)

#%%
# =============================================================================
# System coupling and solution
# =============================================================================
D12u = D12.copy()
D12u = sp.lil_matrix(D12u)
D12u[boundaries,:] = 0
F12u = F12.copy()
F12u[boundaries] = 0

D21v = D21.copy()
D21v = sp.lil_matrix(D21v)
D21v[boundaries,:] = 0
F21v = F21.copy()
F21v[boundaries] = 0

A = sp.vstack((
    sp.hstack((D11, D12u)),
    sp.hstack((D21v, D22))
))

A = sp.csr_array(A)

F = np.hstack((
    F11 + F12u,
    F21v + F22
))

U = sp.linalg.spsolve(A,F)

#%%
# =============================================================================
# Solution plot
# =============================================================================
u = U[:N]
v = U[N:]
displacement = np.sqrt(u**2 + v**2)

fig = plt.figure(figsize=(9,3))
plt.axis("equal")
plt.title("Displacement")
scat = plt.scatter(
    coords[:,0]+u,
    coords[:,1]+v,
    c=displacement,
    cmap="plasma"
)
fig.colorbar(scat)
plt.savefig("figures/05displacement.jpg", dpi=300)

#%%
plt.show()

#%%
