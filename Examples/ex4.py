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

import GFDMI_GN

# loading mesh from file
import json
with open("Examples/Meshes/mesh4.json","r") as f:
    mesh_data = json.load(f)

for key in mesh_data.keys():
    globals()[key] = np.array(mesh_data[key])

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
    triangles=triangles,
    L=L11,
    source=source_u,
    materials=materials,
    dirichlet_boundaries=u_Dir,
    neumann_boundaries=u_Neu,
    Ln_gen=generate_Ln_for_u_equationA
)
K12, F12 = GFDMI_GN.create_system_K_F(
    p=coords,
    triangles=triangles,
    L=L12,
    source=source_v,
    materials=materials,
    dirichlet_boundaries=v_Dir,
    neumann_boundaries=v_Neu,
    Ln_gen=generate_Ln_for_v_equationA
)
K21, F21 = GFDMI_GN.create_system_K_F(
    p=coords,
    triangles=triangles,
    L=L21,
    source=source_u,
    materials=materials,
    dirichlet_boundaries=u_Dir,
    neumann_boundaries=u_Neu,
    Ln_gen=generate_Ln_for_u_equationB
)
K22, F22 = GFDMI_GN.create_system_K_F(
    p=coords,
    triangles=triangles,
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
N = coords.shape[0]
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
# plt.savefig("figures/05bdisplacement.jpg", dpi=300)

plt.show()