#%% Importing needed libraries
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["figure.autolayout"] = True
import scipy.sparse as sp

from GFDMI import GFDMI_2D_problem as gfdmi

#%% Loading mesh from file
import json
with open('Examples/Meshes/mesh1.json', 'r') as file:
    loaded_data = json.load(file)

for key in loaded_data.keys():
    globals()[key] = np.array(loaded_data[key])

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
flux_difference = lambda p: 0
# solution diference at interface u_{mat0} - u_{mat1} = alpha
solution_difference = lambda p: 0.5


#%% System `KU=F` assembling
problem = gfdmi(coords,triangles,L,source)

problem.add_material('material0', permeability_mat0, interior_material_0_nodes)
problem.add_material('material1', permeability_mat1, interior_material_1_nodes)

problem.add_neumann_boundary('bottom_left', permeability_mat0, bottom_left_nodes, bottom_condition)
problem.add_neumann_boundary('top_left', permeability_mat0, top_left_nodes, top_condition)
problem.add_neumann_boundary('top_right', permeability_mat1, top_right_nodes, top_condition)
problem.add_neumann_boundary('bottom_right', permeability_mat1, bottom_right_nodes, bottom_condition)

problem.add_dirichlet_boundary('left', left_nodes, left_condition)
problem.add_dirichlet_boundary('right', right_nodes, right_condition)

problem.add_interface(
    'interface',
    permeability_mat0,
    permeability_mat1,
    left_interface_nodes,
    right_interface_nodes,
    flux_difference,
    solution_difference,
    interior_material_0_nodes,
    interior_material_1_nodes
)

K,F = problem.create_system_K_F()

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
# plt.savefig("figures/01contourf.jpg", dpi=300)

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
# plt.savefig("figures/01-3d.jpg", dpi=300)

#%%
plt.show()