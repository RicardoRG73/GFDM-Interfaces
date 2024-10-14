#%% importing needed libraries
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

plt.style.use("seaborn-v0_8")
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True

import GFDMI

from scipy.integrate import solve_ivp

#%% creating geometry object
g = cfg.Geometry()

# points
g.point([3, 0]) #0
g.point([21, 0]) #1
g.point([29, 0]) #2
g.point([40, 0]) #3
g.point([47, 0]) #4
g.point([27, 10]) #5
g.point([23, 10]) #6
g.point([19, 8]) #7

# custom markers
left_b=108
right_b=100
bottom_b=101
top_b=102

# lines with markers
g.spline([0, 1], marker=bottom_b) #0
g.spline([1, 2], marker=bottom_b) #1
g.spline([2, 3], marker=bottom_b) #2
g.spline([3, 4], marker=right_b,el_on_curve=15) #3
g.spline([4, 5], marker=top_b) #4
g.spline([5, 6], marker=top_b) #5
g.spline([6, 7], marker=top_b, el_on_curve=10) #6
g.spline([7, 0], marker=left_b,el_on_curve=15) #7
g.spline([1, 6]) #8
g.spline([2, 5]) #9

# material markers
Rockfill=100
Clay=102
 
# surfaces with markers
g.surface([0, 8, 6, 7],marker=Rockfill) #0
g.surface([1, 9, 5, 8],marker=Clay) #1
g.surface([2,3,4,9],marker=Rockfill) #2

#%% ploting geometry
cfv.figure(fig_size=(6,4))
cfv.title('Geometry')
cfv.draw_geometry(g)

#%% creating mesh object
mesh = cfm.GmshMesh(g)

mesh.el_type = 2  #2: 3-nodes triangle; 9: 6-nodes triangle
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


#%% identification of boundaries
left_nodes = np.asarray(bdofs[left_b])
right_nodes = np.asarray(bdofs[right_b])
bottom_nodes = np.asarray(bdofs[bottom_b])
bottom_nodes = np.setdiff1d(bottom_nodes , np.intersect1d(bottom_nodes,left_nodes))
bottom_nodes = np.setdiff1d(bottom_nodes , np.intersect1d(bottom_nodes,right_nodes))
top_nodes = np.asarray(bdofs[top_b])
top_nodes = np.setdiff1d(top_nodes , np.intersect1d(top_nodes,left_nodes))
top_nodes = np.setdiff1d(top_nodes , np.intersect1d(top_nodes,right_nodes))


boundaries = np.hstack((left_nodes, right_nodes, bottom_nodes, top_nodes))

interior = np.setdiff1d(
    np.arange(coords.shape[0]),
    boundaries
)

#%% plotting nodes by color
plt.figure()
opacity = 0.5
node_size = 40

# plt.scatter(coords[interior,0], coords[interior,1], label="interior", alpha=opacity)
plt.scatter(coords[left_nodes,0], coords[left_nodes,1], label="bizq", alpha=opacity)
plt.scatter(coords[right_nodes,0], coords[right_nodes,1], label="bder", alpha=opacity)
plt.scatter(coords[bottom_nodes,0], coords[bottom_nodes,1], label="bnb", alpha=opacity)
plt.scatter(coords[top_nodes,0], coords[top_nodes,1], label="bnt4", alpha=opacity)

plt.axis("equal")
plt.legend()

#%% plorblem parameters
L = np.array([0,0,0,1,0,1])
k = lambda p: 1
source = lambda p: 0
neumann_cond = lambda p: 0
left_dirichlet = lambda p: 8
right_dirichlet = lambda p: 0

#%% boundary condition assembled as dictionaries
material = {}
material["0"] = [k, interior]

neumann = {}
neumann["0"] = [k, bottom_nodes, neumann_cond]
neumann["1"] = [k, top_nodes, neumann_cond]

dirichlet = {}
dirichlet["izq"] = [left_nodes, left_dirichlet]
dirichlet["der"] = [right_nodes, right_dirichlet]

interfaces = {}

#%% system KU=F assembling
K,F = GFDMI.create_system_K_F(
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

#%% 3D plot
plt.figure()
ax = plt.axes(projection="3d")
ax.plot_trisurf(
    coords[:,0],
    coords[:,1],
    U,
    cmap="plasma"
)

#%% contourf plot
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

#%%
# =============================================================================
# Difusion equation
# \nabla^2 u + f = du/dt
# =============================================================================
t = [0,200]
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
