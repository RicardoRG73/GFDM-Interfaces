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
g.point([0,0])      # 0
g.point([100,0])      # 1
g.point([100,20])      # 2
g.point([0,20])      # 3

# lines
left = 10
right = 11
top = 12
bottom = 15

g.line([0,1], marker=bottom)    # 0
g.line([1,2], marker=right)     # 1
g.line([2,3], marker=top)       # 2
g.line([3,0], marker=left)      # 3


# surfaces
mat0 = 0
g.surface([0,1,2,3], marker=mat0)


# plotting
cfv.figure()
cfv.title('g')
cfv.draw_geometry(g, draw_axis=True)

#%%
# =============================================================================
# Mesh
# =============================================================================
mesh = cfm.GmshMesh(g,el_size_factor=3)

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

#%%
# =============================================================================
# Problem parameters
# =============================================================================
k = lambda p: 1
f = lambda p: 0

E = 100000
nu = 0.2

alpha = E * (1 - nu) / (1 + nu) / (1 - 2*nu)

a = np.array([
    [1, nu/(1-nu), 0],
    [nu/(1-nu), 1, 0],
    [0, 0, (1-2*nu)/2/(1-nu)]
])

a *= alpha

L11 = [0,0,0,a[0,0],a[0,2]+a[2,0],a[2,2]]
L12 = [0,0,0,a[0,2],a[0,1]+a[2,2],a[2,1]]
L21 = [0,0,0,a[2,0],a[1,0]+a[2,2],a[1,2]]
L22 = [0,0,0,a[2,2],a[1,2]+a[2,1],a[1,1]]

#%%
# =============================================================================
# Boundary conditions
# =============================================================================
fDir = lambda p: 0
fNeu = lambda p: 0
fNeu_load = lambda p: 3e-10

#%%
# =============================================================================
# Discretization with GFDM
# =============================================================================
from GFDMI import create_system_K_F

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
vNeu["right"] = [k, nodesr, fNeu]
vNeu["top"] = [k, nodest, fNeu_load]

D11, F11 = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L11,
    source=f,
    materials=materials,
    dirichlet_boundaries=uDir,
    neumann_boundaries=uNeu
)
D12, F12 = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L12,
    source=f,
    materials=materials,
    dirichlet_boundaries=vDir,
    neumann_boundaries=vNeu
)
D21, F21 = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L21,
    source=f,
    materials=materials,
    dirichlet_boundaries=uDir,
    neumann_boundaries=uNeu
)
D22, F22 = create_system_K_F(
    p=coords,
    triangles=faces,
    L=L12,
    source=f,
    materials=materials,
    dirichlet_boundaries=vDir,
    neumann_boundaries=vNeu
)

#%%
# =============================================================================
# System coupling and solution
# =============================================================================
D12u = D12.copy()
D12u = sp.lil_matrix(D12u)
D12u[Boundaries,:] = 0
F12u = F12.copy()
F12u[Boundaries] = 0

D21v = D21.copy()
D21v = sp.lil_matrix(D21v)
D21v[Boundaries,:] = 0
F21v = F21.copy()
F21v[Boundaries] = 0

A = sp.vstack((
    sp.hstack((D11, D12u)),
    sp.hstack((D21v, D22))
))

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
    coords[:,0]+u,
    coords[:,1]+v,
    c=displacement,
    cmap=color_map
)
fig.colorbar(scat)

#%%
plt.show()

#%%
