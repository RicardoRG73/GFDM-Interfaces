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
permeability_mat1 = lambda p: 1
source = lambda p: 0
left_condition = lambda p: 1 - 0.5*p[1]**2
right_condition = lambda p: 0
bottom_condition = lambda p: 0
top_condition = lambda p: 0
# flux difference at interface du/dn|_{mat0} - du/dn|_{mat1} = beta
flux_difference = lambda p: 0
# solution diference at interface u_{mat0} - u_{mat1} = alpha
solution_difference = lambda p: 0.0


#%% problem definition
problem = gfdmi(coords,triangles,L,source)

problem.add_material('material0', permeability_mat0, interior_material_0_nodes)
problem.add_material('material1', permeability_mat1, interior_material_1_nodes)

# problem.add_neumann_boundary('bottom_left', permeability_mat0, bottom_left_nodes, bottom_condition)
# problem.add_neumann_boundary('top_left', permeability_mat0, top_left_nodes, top_condition)
# problem.add_neumann_boundary('top_right', permeability_mat1, top_right_nodes, top_condition)
# problem.add_neumann_boundary('bottom_right', permeability_mat1, bottom_right_nodes, bottom_condition)

problem.add_dirichlet_boundary('left', left_nodes, right_condition)
problem.add_dirichlet_boundary('right', right_nodes, right_condition)
problem.add_dirichlet_boundary('bottom_left', bottom_left_nodes, right_condition)
problem.add_dirichlet_boundary('top_left', top_left_nodes, right_condition)
problem.add_dirichlet_boundary('top_right', top_right_nodes, right_condition)
problem.add_dirichlet_boundary('bottom_right', bottom_right_nodes, right_condition)

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

#%% System `KU=F` assembling
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



#%% Wave equation
# time integration parameters
tspan = [0,0.5]
dt = 0.01
steps = int((tspan[1]-tspan[0])/dt) + 1

# Mass matrix
M = sp.csr_matrix(np.eye(coords.shape[0]))
# Damping matrix
C = sp.csr_matrix(np.zeros((coords.shape[0], coords.shape[0])))

# initial conditions
U = np.zeros((coords.shape[0], steps))
Ut = np.zeros((coords.shape[0], steps))
Utt = np.zeros((coords.shape[0], steps))

# displacement
U[:,0] = np.sin(np.pi*coords[:,0]) * np.sin(np.pi*coords[:,1])
# velocity
Ut[:,0] = np.zeros(coords.shape[0])
# acceleration
Utt[:,0] = sp.linalg.spsolve(M, F - C @ Ut[:,0] - K @ U[:,0])

# Newmark parameters
delta = 0.5
alpha = 0.25

# Newmark coefficients
a0 = 1 / (alpha*dt**2)
a1 = delta / (alpha*dt)
a2 = 1 / (alpha*dt)
a3 = 1 / (2*alpha) - 1
a4 = delta / alpha - 1
a5 = dt/2 * (delta / alpha - 2)
a6 = dt * (1 - delta)
a7 = delta * dt

# Effective stiffness matrix
K_hat = K + a0*M + a1*C

for n in range(steps-1):
    # Effecive loads
    F_hat = F + M @ (a0*U[:,n] + a2*Ut[:,n] + a3*Utt[:,n]) + C @ (a1*U[:,n] + a4*Ut[:,n] + a5*Utt[:,n])
    # Solve for displacements
    U[:,n+1] = sp.linalg.spsolve(K_hat, F_hat)
    # Calculate accelerations and velocities
    Utt[:,n+1] = a0*(U[:,n+1] - U[:,n]) - a2*Ut[:,n] - a3*Utt[:,n]
    Ut[:,n+1] = Ut[:,n] + a6*Utt[:,n] + a7*Utt[:,n+1]
    
#%% Animation
from matplotlib.animation import FuncAnimation 
fig = plt.figure()

zlims = [-1,1]

ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2, projection="3d")

index = 0
cont = ax1.tricontourf(
    coords[:,0],
    coords[:,1],
    U[:,index],
    cmap="plasma",
    levels=20
)
fig.colorbar(cont)
ax1.set_title("t = %1.2f" %(tspan[0]+dt*index))
ax1.axis("equal")

surf = ax2.plot_trisurf(
    coords[:,0],
    coords[:,1],
    U[:,index],
    cmap="plasma",
    aa=False
)
ax2.set_zlim(zlims[0], zlims[1])
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.set_zlabel("U")
ax2.set_title("3D View")

def update(frame):
    ax1.clear()
    ax2.clear()

    cont = ax1.tricontourf(
        coords[:,0],
        coords[:,1],
        U[:,frame],
        cmap="plasma",
        levels=20
    )
    ax1.set_title("t = %1.2f" %(tspan[0]+frame*dt))
    ax1.axis("equal")

    surf = ax2.plot_trisurf(
        coords[:,0],
        coords[:,1],
        U[:,frame],
        cmap="plasma",
        aa=False
    )
    ax2.view_init(30,-60)
    ax2.set_zlim(zlims[0], zlims[1])
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_zlabel("U")
    ax2.set_title("3D View")
    return cont, surf
ani = FuncAnimation(fig, update, frames=U.shape[1], blit=False, interval=10)
# ani.save("Examples/figures/ex1wave.gif", writer='imagemagick', fps=5)
plt.show()