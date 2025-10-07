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

import GFDMIex02

#%% Loading mesh from file
import json
with open('Examples/Meshes/mesh2.json', 'r') as file:
    loaded_data = json.load(file)

for key in loaded_data.keys():
    globals()[key] = np.array(loaded_data[key])

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
    n = GFDMIex02.normal_vectors(interface_left_nodes, coords)
    i = np.argmin(
        np.sqrt(
            (coords[interface_left_nodes,0]-p[0])**2
            +
            (coords[interface_left_nodes,1]-p[1])**2
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
materials['material_left'] = [permeability_left, omega_plus_nodes]
materials['material_right'] = [permeability_right, omega_minus_nodes]

neumann_boundaries = {}

dirichlet_boundaries = {}
dirichlet_boundaries["dirichlet"] = [dirichlet_nodes, dirichlet_condition]

interfaces = {}
interfaces["interface0"] = [permeability_left, permeability_right, interface_left_nodes, interface_right_nodes, beta, alpha, omega_plus_nodes, omega_minus_nodes]


#%% Assembling system `KU=F`
from GFDMIex02 import create_system_K_F
K,F = create_system_K_F(
    p=coords,
    triangles=triangles,
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
# plt.savefig("figures/03contourf.jpg", dpi=300)

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
# plt.savefig("figures/03-3d.jpg", dpi=300)

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