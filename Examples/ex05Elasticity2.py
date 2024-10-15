#%%
# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-talk"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

import GFDMI_GN

#%%
# =============================================================================
# Geometry
# =============================================================================
g = cfg.Geometry()

# points
omega_length = 4
omega_height = 1
g.point([0, 0])      # 0
g.point([omega_length, 0])      # 1
g.point([omega_length, omega_height])      # 2
g.point([0, omega_height])      # 3

# lines
left = 10
right = 11
top = 12
bottom = 15

g.line([0,1], marker=bottom, el_on_curve=4)    # 0
g.line([1,2], marker=right, el_on_curve=1)     # 1
g.line([2,3], marker=top, el_on_curve=4)       # 2
g.line([3,0], marker=left, el_on_curve=1)      # 3


# surfaces
mat0 = 0
g.struct_surf([0,1,2,3], marker=mat0)


#%% plotting geometry
cfv.figure()
cfv.title('Geometry')
cfv.draw_geometry(g, draw_axis=True)
plt.savefig("figures/05bgeometry.jpg", dpi=300)


#%%
# =============================================================================
# Mesh
# =============================================================================
mesh = cfm.GmshMesh(g,el_size_factor=0.4)

coords, edof, dofs, bdofs, elementmarkers = mesh.create()
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)

#%% plotting mesh
cfv.figure()
cfv.title('Malla $N=%d' %coords.shape[0] +'$')
cfv.draw_mesh(
    coords=coords,
    edof=edof,
    dofs_per_node=mesh.dofs_per_node,
    el_type=mesh.el_type,
    filled=True
)
plt.savefig("figures/05bmesh.jpg", dpi=300)


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
        s=100
)
plt.axis("equal")
plt.legend()
plt.savefig("figures/05bnodes.jpg", dpi=300)

#%%
# =============================================================================
# Problem parameters
# =============================================================================
k = lambda p: 1
source_u = lambda p: 0
source_v = lambda p: 0

# E: young modulus, nu: poisson constant
E = 1e5
nu = 0.3

# PS: plain-strain matrix
alpha = E * (1 - nu) / (1 + nu) / (1 - 2*nu)
PS = np.array([
    [1, nu/(1-nu), 0],
    [nu/(1-nu), 1, 0],
    [0, 0, (1-2*nu)/2/(1-nu)]
])
PS *= alpha

L11 = np.array([0,0,0,PS[0,0],PS[0,2]+PS[2,0],PS[2,2]])
L12 = np.array([0,0,0,PS[0,2],PS[0,1]+PS[2,2],PS[2,1]])
L21 = np.array([0,0,0,PS[2,0],PS[1,0]+PS[2,2],PS[1,2]])
L22 = np.array([0,0,0,PS[2,2],PS[1,2]+PS[2,1],PS[1,1]])

#%%
# =============================================================================
# Boundary conditions
# =============================================================================
dirichlet_condition = lambda p: 0
neumann_condition = lambda p: 0
neumann_load = lambda p: 10000

#%%
# =============================================================================
# Discretization with GFDM
# =============================================================================
def generate_Ln_for_u_equationA(p,b,i):
    n = GFDMI_GN.normal_vectors(b,p)
    ni = n[b==i][0]
    nx, ny = ni
    term1 = nx * PS[0,0] + ny * PS[2,0]
    term2 = nx * PS[0,2] + ny * PS[2,2]
    Ln = np.array([0,term1,term2,0,0,0])
    return Ln

def generate_Ln_for_v_equationA(p,b,i):
    n = GFDMI_GN.normal_vectors(b,p)
    ni = n[b==i][0]
    nx, ny = ni
    term1 = nx * PS[0,1] + ny * PS[2,1]
    term2 = nx * PS[0,2] + ny * PS[2,2]
    Ln = np.array([0,term1,term2,0,0,0])
    return Ln

def generate_Ln_for_u_equationB(p,b,i):
    n = GFDMI_GN.normal_vectors(b,p)
    ni = n[b==i][0]
    nx, ny = ni
    term3 = ny * PS[1,0] + nx * PS[2,0]
    term4 = ny * PS[1,2] + nx * PS[2,2]
    Ln = np.array([0,term3,term4,0,0,0])
    return Ln

def generate_Ln_for_v_equationB(p,b,i):
    n = GFDMI_GN.normal_vectors(b,p)
    ni = n[b==i][0]
    nx, ny = ni
    term3 = ny * PS[1,1] + nx * PS[2,1]
    term4 = ny * PS[1,2] + nx * PS[2,2]
    Ln = np.array([0,term3,term4,0,0,0])
    return Ln

def generate_Ln_laplacian(p,b,i):
    n = GFDMI_GN.normal_vectors(b,p)
    ni = n[b==i][0]
    nx, ny = ni
    Ln = np.array([0,0,0,2,0,2])
    return Ln

# boundary condition assembled as dictionaries
materials = {}
materials["0"] = [k, interior_nodes]

u_Dir = {}
u_Dir["left"] = [left_nodes, dirichlet_condition]

u_Neu = {}
u_Neu["bottom"] = [k, bottom_nodes, neumann_condition]
u_Neu["right"] = [k, right_nodes, neumann_load]
u_Neu["top"] = [k, top_nodes, neumann_condition]

v_Dir = {}
v_Dir["left"] = [left_nodes, dirichlet_condition]

v_Neu = {}
v_Neu["bottom"] = [k, bottom_nodes, neumann_condition]
v_Neu["right"] = [k, right_nodes, neumann_condition]
v_Neu["top"] = [k, top_nodes, neumann_condition]

BNeum = np.hstack((bottom_nodes, right_nodes, top_nodes)) 

#%% system KU=F assembling
K11, F11 = GFDMI_GN.create_system_K_F(
    p=coords,
    triangles=faces,
    L=L11,
    source=source_u,
    materials=materials,
    dirichlet_boundaries=u_Dir,
    neumann_boundaries=u_Neu,
    Ln_gen=generate_Ln_for_u_equationA
)
K12, F12 = GFDMI_GN.create_system_K_F(
    p=coords,
    triangles=faces,
    L=L12,
    source=source_v,
    materials=materials,
    dirichlet_boundaries=v_Dir,
    neumann_boundaries=v_Neu,
    Ln_gen=generate_Ln_for_v_equationA
)
K21, F21 = GFDMI_GN.create_system_K_F(
    p=coords,
    triangles=faces,
    L=L21,
    source=source_u,
    materials=materials,
    dirichlet_boundaries=u_Dir,
    neumann_boundaries=u_Neu,
    Ln_gen=generate_Ln_for_u_equationB
)
K22, F22 = GFDMI_GN.create_system_K_F(
    p=coords,
    triangles=faces,
    L=L12,
    source=source_v,
    materials=materials,
    dirichlet_boundaries=v_Dir,
    neumann_boundaries=v_Neu,
    Ln_gen=generate_Ln_for_v_equationB
)

#%%
# =============================================================================
# System coupling and solution
# =============================================================================
K12u = K12.copy()
K12u = sp.lil_matrix(K12u)
K12u[boundaries,:] = 0
F12u = F12.copy()
F12u[boundaries] = 0

K21v = K21.copy()
K21v = sp.lil_matrix(K21v)
K21v[boundaries,:] = 0
F21v = F21.copy()
F21v[boundaries] = 0

K = sp.vstack((
    sp.hstack((K11, K12u)),
    sp.hstack((K21v, K22))
))

K = sp.csr_array(K)

F = np.hstack((
    F11 + F12u,
    F21v + F22
))

U = sp.linalg.spsolve(K,F)

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
    coords[:,0] + u,
    coords[:,1] + v,
    c=displacement,
    cmap="plasma"
)
fig.colorbar(scat)
plt.plot(
    [0,omega_length,omega_length,0,0],
    [0,0,omega_height,omega_height,0],
    alpha=0.25,
    color="k"
)
plt.savefig("figures/05bdisplacement.jpg", dpi=300)

#%%
plt.show()

#%%
