#%%
# =============================================================================
# Importing needed libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

plt.style.use("seaborn-v0_8")

from scipy.integrate import solve_ivp

from GFDMI import GFDMI_2D_problem as gfdmi


# loading mesh data
import json
with open("Examples/Meshes/mesh3.json","r") as f:
    mesh_data = json.load(f)

for key in mesh_data.keys():
    globals()[key] = np.array(mesh_data[key])

#%%
# =============================================================================
# Problem parameters
# =============================================================================
L = np.array([0,0,0,1,0,1])
kr = lambda p: 1
kc = lambda p: 1e-1
source = lambda p: 0
neumann_cond = lambda p: 0
left_dirichlet = lambda p: 8
right_dirichlet = lambda p: 0
beta = lambda p: 0

#%%
# =============================================================================
# Assembling and solving system KU=F
# =============================================================================
# material = {}
# material["rock"] = [kr, rock_nodes]
# material["clay"] = [kc, clay_nodes]

# neumann = {}
# neumann["0"] = [kr, bottom_nodes, neumann_cond]
# neumann["1"] = [kr, top_nodes, neumann_cond]

# dirichlet = {}
# dirichlet["izq"] = [left_nodes, left_dirichlet]
# dirichlet["der"] = [right_nodes, right_dirichlet]

# interfaces = {}
# interfaces["A"] = [kr, kc, left_interface_nodes, beta, rock_nodes, clay_nodes]
# interfaces["B"] = [kc, kr, right_interface_nodes, beta, clay_nodes, rock_nodes]

problem = gfdmi(
    coords,
    triangles,
    L,
    source
)

problem.add_material("rock", kr, rock_nodes)
problem.add_material("clay", kc, clay_nodes)

problem.add_neumann_boundary("bottom", kr, bottom_nodes, neumann_cond)
problem.add_neumann_boundary("top", kr, top_nodes, neumann_cond)

problem.add_dirichlet_boundary("left", left_nodes, left_dirichlet)
problem.add_dirichlet_boundary("right", right_nodes, right_dirichlet)

problem.add_interface(
    "left_interface",
    kr,
    kc,
    left_interface_nodes,
    None,
    beta,
    None,
    rock_nodes,
    clay_nodes
)

problem.add_interface(
    "right_interface",
    kc,
    kr,
    right_interface_nodes,
    None,
    beta,
    None,
    clay_nodes,
    rock_nodes
)


#%% system KU=F assembling
K,F = problem.create_system_K_F_cont_U()

#%% system KU=F solution
U = sp.linalg.spsolve(K,F)

#%%
# =============================================================================
# Plotting U
# =============================================================================
# 3D
plt.figure()
ax = plt.axes(projection="3d")
ax.plot_trisurf(
    coords[:,0],
    coords[:,1],
    U,
    cmap="plasma"
)
plt.title(r"Stationary solution $\nabla^2 u = 0$")
# plt.savefig("figures/04b-3d.jpg", dpi=300)

#%% contourf
plt.figure()
plt.tricontourf(
    coords[:,0],
    coords[:,1],
    U,
    cmap="plasma",
    levels=20
)
plt.axis("equal")
plt.colorbar()
# line h=0
plt.tricontour(
    coords[:,0],
    coords[:,1],
    (U - coords[:,1])*9.81,
    levels=[0.0],
    colors="b"
)
plt.title(r"Stationary solution $\nabla^2 u = 0$")
# plt.savefig("figures/04bcontourf.jpg", dpi=300)

#%%
# =============================================================================
# Difusion equation
# \nabla^2 u + f = du/dt
# =============================================================================
t = [0,80]
fun = lambda t,U: K@U - F
U0 = np.zeros(coords.shape[0])
U0[left_nodes] = 8
U0[right_nodes] = 0

#%% initial condition plot
plt.figure()
ax = plt.axes(projection="3d")
ax.plot_trisurf(
    coords[:,0],
    coords[:,1],
    U0,
    cmap="plasma"
)
ax.set_title("Initial Condition $U_0$")

#%% solution
sol = solve_ivp(fun, t, U0)

U_difussion = sol.y

#%% plots
fig = plt.figure()

final_index = sol.t.shape[0] - 1
times_index = [0, final_index//10, final_index//3, final_index]

for i,t_i in enumerate(times_index):
    ax = plt.subplot(2,2,i+1)
    ax.tricontourf(
        coords[:,0],
        coords[:,1],
        U_difussion[:,t_i],
        cmap="plasma",
        levels=20
    )
    ax.tricontour(
        coords[:,0],
        coords[:,1],
        (U_difussion[:,t_i] - coords[:,1])*9.81,
        levels=[0.0],
        colors="k",
        linewidths=0.5
    )
    ax.axis("equal")
    ax.set_title("$t = %1.2f$" %sol.t[t_i])
# plt.savefig("figures/04bdiffussion.jpg", dpi=300)

#%% 3d plot at final time
plt.figure()
ax = plt.axes(projection="3d")
ax.plot_trisurf(
    coords[:,0],
    coords[:,1],
    U_difussion[:,final_index],
    cmap="plasma"
)
ax.set_title("Solution $U$ at time $t=%1.2f$" %sol.t[-1])

# condition number
print("\n\n Condition number cond(K): %1.3e" %np.linalg.cond(K.toarray()))

plt.show()
# %%
