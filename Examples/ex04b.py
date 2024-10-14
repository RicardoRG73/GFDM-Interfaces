#%%
# =============================================================================
# Importing needed libraries
# =============================================================================
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

plt.style.use("seaborn-v0_8")

import GFDMI

from scipy.integrate import solve_ivp

#%%
# =============================================================================
# Geometry
# =============================================================================
g = cfg.Geometry()  # Create a GeoData object that holds the geometry.
g.point([3, 0]) #0
g.point([21, 0]) #1
g.point([29, 0]) #2
g.point([40, 0]) #3
g.point([47, 0]) #4
g.point([27, 10]) #5
g.point([23, 10]) #6
g.point([19, 8]) #7

left_b=108
right_b=100
bottom_b=101
top_b=102

left_interface = 110
right_interface = 111

g.spline([0, 1], marker=bottom_b) #0
g.spline([1, 2], marker=bottom_b) #1
g.spline([2, 3], marker=bottom_b) #2
g.spline([3, 4], marker=right_b,el_on_curve=15) #3
g.spline([4, 5], marker=top_b) #4
g.spline([5, 6], marker=top_b) #5
g.spline([6, 7], marker=top_b, el_on_curve=10) #6
g.spline([7, 0], marker=left_b,el_on_curve=15) #7
g.spline([1, 6], marker=left_interface) #8
g.spline([2, 5], marker=right_interface) #9

Rockfill=100
Clay=102
 
g.surface([0, 8, 6, 7],marker=Rockfill) #0
g.surface([1, 9, 5, 8],marker=Clay) #1
g.surface([2,3,4,9],marker=Rockfill) #2

#%% geometry plot
cfv.figure(fig_size=(6,4))
cfv.title('Geometry')
cfv.draw_geometry(g)

#%% mesh creation
# =============================================================================
# Mesh
# =============================================================================
mesh = cfm.GmshMesh(g)

mesh.el_type = 2  #2= triangulo de 3 nodos 9= triangulo de 6 nodos
mesh.dofs_per_node = 1  # Degrees of freedom per node.
mesh.el_size_factor = 2.0  # Factor that changes element sizes.

coords, edof, dofs, bdofs, element_markers = mesh.create()

# mesh conditioning
nodes_in_triangle = edof.shape[1]
triangles = np.zeros(edof.shape, dtype=int)
for i,elem in enumerate(edof):
    triangles[i,:] = elem[1],elem[0],elem[2]
triangles = triangles-1
bdofs = {frontera : np.array(bdofs[frontera])-1 for frontera in bdofs}

#%% mesh plot
cfv.figure(fig_size=(6,4))
cfv.title('Mesh')
cfv.draw_mesh(coords=coords, edof=edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, filled=True)

#%%
# =============================================================================
# Nodes index
# =============================================================================
left_nodes = np.asarray(bdofs[left_b])
right_nodes = np.asarray(bdofs[right_b])
bottom_nodes = np.asarray(bdofs[bottom_b])
bottom_nodes = np.setdiff1d(bottom_nodes , np.intersect1d(bottom_nodes,left_nodes))
bottom_nodes = np.setdiff1d(bottom_nodes , np.intersect1d(bottom_nodes,right_nodes))
top_nodes = np.asarray(bdofs[top_b])
top_nodes = np.setdiff1d(top_nodes , np.intersect1d(top_nodes,left_nodes))
top_nodes = np.setdiff1d(top_nodes , np.intersect1d(top_nodes,right_nodes))
left_interface_nodes = np.asarray(bdofs[left_interface])
left_interface_nodes = np.setdiff1d(left_interface_nodes, np.intersect1d(left_interface_nodes, top_nodes))
left_interface_nodes = np.setdiff1d(left_interface_nodes, np.intersect1d(left_interface_nodes, bottom_nodes))
right_interface_nodes = np.asarray(bdofs[right_interface])
right_interface_nodes = np.setdiff1d(right_interface_nodes, np.intersect1d(right_interface_nodes, top_nodes))
right_interface_nodes = np.setdiff1d(right_interface_nodes, np.intersect1d(right_interface_nodes, bottom_nodes))

boundaries = np.hstack((
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_nodes,
    left_interface_nodes,
    right_interface_nodes
))

element_markers = np.array(element_markers)

rock_nodes = triangles[element_markers == Rockfill]
rock_nodes = rock_nodes.flatten()
rock_nodes = np.setdiff1d(rock_nodes, boundaries)

clay_nodes = triangles[element_markers == Clay]
clay_nodes = clay_nodes.flatten()
clay_nodes = np.setdiff1d(clay_nodes, boundaries)

#%% plotting nodes by color
plt.figure()
nodes_to_plot = (
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_nodes,
    left_interface_nodes,
    right_interface_nodes,
    rock_nodes,
    clay_nodes
)
labels = (
    "Left",
    "Right",
    "Bottom",
    "Top",
    "Left Interface",
    "Right Interface",
    "Rock",
    "Clay"
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

# =============================================================================
# Assembling and solving system KU=F
# =============================================================================
from GFDMI import create_system_K_F_cont_U

material = {}
material["rock"] = [kr, rock_nodes]
material["clay"] = [kc, clay_nodes]

neumann = {}
neumann["0"] = [kr, bottom_nodes, neumann_cond]
neumann["1"] = [kr, top_nodes, neumann_cond]

dirichlet = {}
dirichlet["izq"] = [left_nodes, left_dirichlet]
dirichlet["der"] = [right_nodes, right_dirichlet]

interfaces = {}
interfaces["A"] = [kr, kc, left_interface_nodes, beta, rock_nodes, clay_nodes]
interfaces["B"] = [kc, kr, right_interface_nodes, beta, clay_nodes, rock_nodes]

#%% system KU=F assembling
K,F = GFDMI.create_system_K_F_cont_U(
    p=coords,
    triangles=triangles,
    L=L,
    source=source,
    materials=material,
    neumann_boundaries=neumann,
    dirichlet_boundaries=dirichlet,
    interfaces = interfaces
)

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

# contourf
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
