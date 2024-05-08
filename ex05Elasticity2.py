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
color_map = "plasma"

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

#%%
# =============================================================================
# Geometry
# =============================================================================
g = cfg.Geometry()

# points
L = 4
H = 1
g.point([0,0])      # 0
g.point([L,0])      # 1
g.point([L,H])      # 2
g.point([0,H])      # 3

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


# plotting
cfv.figure()
cfv.title('g')
cfv.draw_geometry(g, draw_axis=True)

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

# plotting
cfv.figure()
cfv.title('Malla $N=%d' %coords.shape[0] +'$')
cfv.draw_mesh(
    coords=coords,
    edof=edof,
    dofs_per_node=mesh.dofs_per_node,
    el_type=mesh.el_type,
    filled=True
)

#%%
# =============================================================================
# Nodes identification by color
# nodesl: left
# nodesr: right
# nodesb: bottom
# nodest: top
# =============================================================================
corners = np.array([0,1,2,3])

nodesl = np.asarray(bdofs[left]) - 1
nodesl = np.setdiff1d(nodesl, corners)
nodesl = np.hstack((nodesl, [0,3]))

nodesr = np.asarray(bdofs[right]) - 1
nodesr = np.setdiff1d(nodesr, corners)
nodesr = np.hstack((nodesr, [1,2]))

nodesb = np.asarray(bdofs[bottom]) - 1
nodesb = np.setdiff1d(nodesb, corners)

nodest = np.asarray(bdofs[top]) - 1
nodest = np.setdiff1d(nodest, corners)

boundaries = (nodesl, nodesr, nodesb, nodest, corners)
Boundaries = np.hstack(boundaries)

N = coords.shape[0]
interior = np.setdiff1d(np.arange(N), Boundaries)

plt.figure(figsize=(9,3))
plt.axis("equal")
s = 30
plt.scatter(coords[interior,0], coords[interior,1], label="interior", s=s)
plt.scatter(coords[nodesb,0], coords[nodesb,1], label="bottom", s=s)
plt.scatter(coords[nodesr,0], coords[nodesr,1], label="right", s=s)
plt.scatter(coords[nodest,0], coords[nodest,1], label="top", s=s)
plt.scatter(coords[nodesl,0], coords[nodesl,1], label="left", s=s)
plt.legend(loc="center")
for i in range(N):
    plt.text(coords[i,0],coords[i,1],s=str(i))

#%%
# =============================================================================
# Problem parameters
# =============================================================================
k = lambda p: 1
fu = lambda p: 0
fv = lambda p: 0

E = 1e5
nu = 0.3

alpha = E * (1 - nu) / (1 + nu) / (1 - 2*nu)

D = np.array([
    [1, nu/(1-nu), 0],
    [nu/(1-nu), 1, 0],
    [0, 0, (1-2*nu)/2/(1-nu)]
])

D *= alpha

L11 = np.array([0,0,0,D[0,0],D[0,2]+D[2,0],D[2,2]])
L12 = np.array([0,0,0,D[0,2],D[0,1]+D[2,2],D[2,1]])
L21 = np.array([0,0,0,D[2,0],D[1,0]+D[2,2],D[1,2]])
L22 = np.array([0,0,0,D[2,2],D[1,2]+D[2,1],D[1,1]])

#%%
# =============================================================================
# Boundary conditions
# =============================================================================
fDir = lambda p: 0
fNeu = lambda p: 0
fNeu_load = lambda p: 500

#%%
# =============================================================================
# Discretization with GFDM
# =============================================================================
from GFDMI_GN import create_system_K_F

from GFDMI_GN import normal_vectors

def Ln_gen_u_A(p,b,i):
    n = normal_vectors(b,p)
    ni = n[b==i][0]
    nx, ny = ni
    term1 = nx * D[0,0] + ny * D[2,0]
    term2 = nx * D[0,2] + ny * D[2,2]
    Ln = np.array([0,term1,term2,0,0,0])
    return Ln

def Ln_gen_v_A(p,b,i):
    n = normal_vectors(b,p)
    ni = n[b==i][0]
    nx, ny = ni
    term1 = nx * D[0,1] + ny * D[2,1]
    term2 = nx * D[0,2] + ny * D[2,2]
    Ln = np.array([0,term1,term2,0,0,0])
    return Ln

def Ln_gen_u_B(p,b,i):
    n = normal_vectors(b,p)
    ni = n[b==i][0]
    nx, ny = ni
    term3 = ny * D[1,0] + nx * D[2,0]
    term4 = ny * D[1,2] + nx * D[2,2]
    Ln = np.array([0,term3,term4,0,0,0])
    return Ln

def Ln_gen_v_B(p,b,i):
    n = normal_vectors(b,p)
    ni = n[b==i][0]
    nx, ny = ni
    term3 = ny * D[1,1] + nx * D[2,1]
    term4 = ny * D[1,2] + nx * D[2,2]
    Ln = np.array([0,term3,term4,0,0,0])
    return Ln

def Ln_gen2(p,b,i):
    n = normal_vectors(b,p)
    ni = n[b==i][0]
    nx, ny = ni
    Ln = np.array([0,0,0,2,0,2])
    return Ln

materials = {}
materials["0"] = [k, interior]

uDir = {}
uDir["left"] = [nodesl, fDir]

uNeu = {}
uNeu["bottom"] = [k, nodesb, fNeu]
uNeu["right"] = [k, nodesr, fNeu]
uNeu["top"] = [k, nodest, fNeu]

vDir = {}
vDir["left"] = [nodesl, fDir]

vNeu = {}
vNeu["bottom"] = [k, nodesb, fNeu]
vNeu["right"] = [k, nodesr, fNeu_load]
vNeu["top"] = [k, nodest, fNeu]

BNeum = np.hstack((nodesb, nodesr, nodest)) 

K11, F11 = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L11,
    source=fu,
    materials=materials,
    dirichlet_boundaries=uDir,
    neumann_boundaries=uNeu,
    Ln_gen=Ln_gen_u_A
)
K12, F12 = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L12,
    source=fv,
    materials=materials,
    dirichlet_boundaries=vDir,
    neumann_boundaries=vNeu,
    Ln_gen=Ln_gen_v_A
)
K21, F21 = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L21,
    source=fu,
    materials=materials,
    dirichlet_boundaries=uDir,
    neumann_boundaries=uNeu,
    Ln_gen=Ln_gen_u_B
)
K22, F22 = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L12,
    source=fv,
    materials=materials,
    dirichlet_boundaries=vDir,
    neumann_boundaries=vNeu,
    Ln_gen=Ln_gen_v_B
)

# K11 = sp.lil_matrix(K11)
# K12 = sp.lil_matrix(K12)
# K21 = sp.lil_matrix(K21)
# K22 = sp.lil_matrix(K22)
# for i in BNeum:
#     K11[:,i] += K12[:,i] ;  K12[:,i] = 0;  K12[i,i] = 1
#     K22[:,i] += K21[:,i] ;  K21[:,i] = 0;  K21[:,i] = 1

#%%
# =============================================================================
# System coupling and solution
# =============================================================================
K12u = K12.copy()
K12u = sp.lil_matrix(K12u)
K12u[Boundaries,:] = 0
F12u = F12.copy()
F12u[Boundaries] = 0

K21v = K21.copy()
K21v = sp.lil_matrix(K21v)
K21v[Boundaries,:] = 0
F21v = F21.copy()
F21v[Boundaries] = 0

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
cont = plt.tricontourf(
    coords[:,0],
    coords[:,1],
    displacement,
    cmap=color_map,
    levels=20
)
fig.colorbar(cont)

fig = plt.figure(figsize=(9,3))
plt.axis("equal")
plt.title("Displacement")
scat = plt.scatter(
    coords[:,0] + u,#/np.max(np.abs(u))/2,
    coords[:,1] + v,#/np.max(np.abs(v))/2,
    c=displacement,
    cmap=color_map
)
fig.colorbar(scat)
plt.plot([0,L,L,0,0],[0,0,H,H,0],alpha=0.5,color="k")

#%%
plt.show()

#%%
