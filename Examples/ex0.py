import json
import numpy as np
import scipy.sparse as sp

import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["figure.autolayout"] = True

from GFDMI import GFDMI_2D_problem as gfdmi

with open('Examples/Meshes/mesh0.json', 'r') as file:
    loaded_data = json.load(file)

for key in loaded_data.keys():
    globals()[key] = np.array(loaded_data[key])

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

# problem definition
problem = gfdmi(coords,triangles,L,source)

problem.add_material('0', permeability, interior_nodes)

problem.add_neumann_boundary('right', permeability, right_nodes, right_condition)

problem.add_dirichlet_boundary('left', left_nodes, left_condition)
problem.add_dirichlet_boundary('top', top_nodes, top_condition)
problem.add_dirichlet_boundary('bottom', bottom_nodes, bottom_condition)


#%% System `KU=F` assembling
K,F = problem.create_system_K_F()

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