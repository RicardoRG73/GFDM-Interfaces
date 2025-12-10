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
source = lambda p: 0
left_condition = lambda p: 0
right_condition = lambda p: 0
bottom_condition = lambda p: 0
top_condition = lambda p: 0

# problem definition
problem = gfdmi(coords,triangles,L,source)

problem.add_material('0', permeability, interior_nodes)

problem.add_dirichlet_boundary('right', right_nodes, right_condition)

problem.add_dirichlet_boundary('left', left_nodes, left_condition)
problem.add_dirichlet_boundary('top', top_nodes, top_condition)
problem.add_dirichlet_boundary('bottom', bottom_nodes, bottom_condition)


#%% System `KU=F` assembling
K,F = problem.create_system_K_F()

#%% Solving wave equation
# time integration parameters
tspan = [0,0.5]
dt = 0.01
steps = int((tspan[1]-tspan[0])/dt) + 1

# Mass matrix
M = sp.csr_matrix(np.eye(coords.shape[0]))
# Damping matrix
C = sp.csr_matrix(np.zeros((coords.shape[0], coords.shape[0])))


# -- initial conditions --
U = np.zeros((coords.shape[0], steps))
Ut = np.zeros((coords.shape[0], steps))
Utt = np.zeros((coords.shape[0], steps))

# displacement
U[:,0] = (coords[:,0]<=1)*np.sin(np.pi*coords[:,1])*np.sin(np.pi*coords[:,0])
# velocity
Ut[:,0] = np.zeros(coords.shape[0])
# acceleration
Utt[:,0] = sp.linalg.spsolve(M, F - C @ Ut[:,0] - K @ U[:,0])

# -- Newmark method --
# parameters
delta = 0.5
alpha = 0.25

# coefficients
a0 = 1 / (alpha*dt**2)
a1 = delta / (alpha*dt)
a2 = 1 / (alpha*dt)
a3 = 1 / (2*alpha) - 1
a4 = delta / alpha - 1
a5 = dt/2 * (delta / alpha - 2)
a6 = dt * (1 - delta)
a7 = delta * dt

# effective stiffness matrix
K_hat = K + a0*M + a1*C

# time integration loop
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
#ani.save("Examples/figures/ex0wave.gif", writer='imagemagick', fps=5)
plt.show()