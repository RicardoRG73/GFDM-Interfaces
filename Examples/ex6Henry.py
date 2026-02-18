#%%
# -- libraries --
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-paper"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1
mapa_de_color = "plasma"

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

from GFDMI import GFDMI_2D_problem as gfdmi

#%%
# -- geometry --

geometria = cfg.Geometry()

# points
geometria.point([0,0])      # 0
geometria.point([2,0])      # 1
geometria.point([2,1])      # 2
geometria.point([0,1])      # 3

# lines
left = 10
right = 11
top = 12
bottom = 13

geometria.line([0,1], marker=bottom)    # 0
geometria.line([1,2], marker=right)     # 1
geometria.line([2,3], marker=top)       # 2
geometria.line([3,0], marker=left)      # 3

# surfaces
mat0 = 0
geometria.surface([0,1,2,3], marker=mat0)

# geometry plot
cfv.figure(fig_size=(8,5))
cfv.title('Geometría')
cfv.draw_geometry(geometria, font_size=16, draw_axis=True)

#%%
# -- mesh --
# | el_size_factor |   N   |
# |     0.1        |  274  |
# |     0.05       |  998  |
# |     0.03       | 2748  |
# |     0.02       | 5969  |

mesh = cfm.GmshMesh(geometria)

mesh.el_type = 2                            # type of element: 2 = triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.1

coords, edof, dofs, bdofs, elementmarkers = mesh.create()   # create the geometry
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)

# mesh plot
plt.figure(figsize=(7,4))
cfv.title('Malla $N=%d' %coords.shape[0] +'$')
cfv.draw_mesh(
    coords=coords,
    edof=edof,
    dofs_per_node=mesh.dofs_per_node,
    el_type=mesh.el_type,
    filled=True
)

#%%
# -- boundary nodes --
left_nodes = np.asarray(bdofs[left]) - 1
left_nodes = np.setdiff1d(left_nodes, [0,3])
right_nodes = np.asarray(bdofs[right]) - 1
right_nodes = np.setdiff1d(right_nodes, [1,2])
bottom_nodes = np.asarray(bdofs[bottom]) - 1
bottom_nodes = np.setdiff1d(bottom_nodes, [0,1])
top_nodes = np.asarray(bdofs[top]) - 1
top_nodes = np.setdiff1d(top_nodes, [2,3])
corner_nodes = np.array([0,1,2,3])

boundary_nodes = np.hstack((
    left_nodes, right_nodes, bottom_nodes, top_nodes, corner_nodes
))
interior_nodes = np.setdiff1d(np.arange(coords.shape[0]) , boundary_nodes)
nodes_to_plot = (interior_nodes, left_nodes, right_nodes, bottom_nodes, top_nodes, corner_nodes)
labels = (
    "Interior",
    "Left Boundary",
    "Right Boundary",
    "Bottom Boundary",
    "Top Boundary",
    "Corners"
)

plt.figure(figsize=(7,4))
for nodes, label in zip(nodes_to_plot, labels):
    plt.scatter(
        coords[nodes, 0],
        coords[nodes, 1],
        label=label,
        s=20,
        alpha=0.5
    )
plt.axis('equal')
plt.legend()

#%%
# -- problem parameters --
# Henry: a=0.2637, b=0.1
# Pinder: a=0.2637, b=0.035
# Modified: a=0.1315, b=0.2

a = 0.2637
b = 0.035
k = lambda p: 1
source = lambda p: 0

#%%
# -- D matrices for flow function \Psi --
Psit = lambda p: 1
Psib = lambda p: 0
Psil = lambda p: 0
Psir = lambda p: 0

L2 = np.array([0,0,0,2,0,2])

problem = gfdmi(coords, faces, L2, source)

problem.material("0", k, interior_nodes)

problem.dirichlet_boundary("top", top_nodes, Psit)
problem.dirichlet_boundary("bottom", bottom_nodes, Psib)
problem.dirichlet_boundary("esquinas_top", [2,3], Psit)
problem.dirichlet_boundary("esquinas_bottom", [0,1], Psib)

problem.neumann_boundary("left", k, left_nodes, Psil)
problem.neumann_boundary("right", k, right_nodes, Psir)

D2psi, F2psi = problem.continuous_discretization()

Lx = np.array([0,1,0,0,0,0])
problem.L = Lx
Dxpsi, Fxpsi = problem.continuous_discretization()

Ly = np.array([0,0,1,0,0,0])
problem.L = Ly
Dypsi, Fypsi = problem.continuous_discretization()

#%%
# -- D matrices for concentration C --
Cl = lambda p: 0
Cr = lambda p: 1
Ct = lambda p: 0
Cb = lambda p: 0

problem = gfdmi(coords, faces, L2, source)

problem.material("0", k, interior_nodes)

problem.dirichlet_boundary("left", left_nodes, Cl)
problem.dirichlet_boundary("right", right_nodes, Cr)
problem.dirichlet_boundary("esquinas_left", [0,3], Cl)
problem.dirichlet_boundary("esquinas_right", [1,2], Cr)

problem.neumann_boundary("top", k, top_nodes, Ct)
problem.neumann_boundary("bottom", k, bottom_nodes, Cb)

D2c, F2c = problem.continuous_discretization()

problem.L = Lx
Dxc, Fxc = problem.continuous_discretization()

problem.L = Ly
Dyc, Fyc = problem.continuous_discretization()

#%%
# -- IVP assembly --
import scipy.sparse as sp

Dxcpsi = Dxc.copy()
Dxcpsi = sp.lil_matrix(Dxcpsi)
Fxcpsi = Fxc.copy()

Dxcpsi[boundary_nodes,:] = 0
Fxcpsi[boundary_nodes] = 0

Dypsic = Dypsi.copy()
Dypsic = sp.lil_matrix(Dypsic)
Fypsic = Fypsi.copy()

Dypsic[boundary_nodes,:] = 0
Fypsic[boundary_nodes] = 0

Dxpsic = Dxpsi.copy()
Dxpsic = sp.lil_matrix(Dxpsic)
Fxpsic = Fxpsi.copy()

Dxpsic[boundary_nodes,:] = 0
Fxpsic[boundary_nodes] = 0

# linear matrix A
N = coords.shape[0]
print("N = ", N)
A = sp.vstack((
    sp.hstack((
        D2psi, -1/a * Dxcpsi
    )),
    sp.hstack((
        np.zeros((N,N)), D2c
    ))
))

# linear vector F
F = np.hstack((
    - F2psi  +  1/a * Fxcpsi,
    - F2c
))

# nonlinear vector B
def B(U):
    term1 = (Dypsic@U[:N] - Fypsic) * (Dxc@U[N:] - Fxc)
    term2 = (Dxpsic@U[:N] - Fxpsic) * (Dyc@U[N:] - Fyc)
    vec2 = -1/b * (term1 - term2)
    vec1 = np.zeros(N)
    vec = np.hstack((vec1, vec2))
    return vec

# right hand side
fun = lambda t,U: A@U + F + B(U)

# initial conditions
C0 = np.zeros(N)
Psi0 = np.zeros(N)
for i in left_nodes:
    C0[i] = Cl(coords[i,:])
    Psi0[i] = Psil(coords[i,:])
for i in right_nodes:
    C0[i] = Cr(coords[i,:])
    Psi0[i] = Psir(coords[i,:])
for i in top_nodes:
    C0[i] = Ct(coords[i,:])
    Psi0[i] = Psit(coords[i,:])
for i in bottom_nodes:
    C0[i] = Cb(coords[i,:])
    Psi0[i] = Psib(coords[i,:])
i = 0
C0[i] = Cl(coords[i,:])
Psi0[i] = Psib(coords[i,:])
i = 1
C0[i] = Cr(coords[i,:])
Psi0[i] = Psib(coords[i,:])
i = 2
C0[i] = Cr(coords[i,:])
Psi0[i] = Psit(coords[i,:])
i = 3
C0[i] = Cl(coords[i,:])
Psi0[i] = Psit(coords[i,:])

U0 = np.hstack((Psi0, C0))

#%%
# IVP solution
t_final = 0.21
tspan = [0, t_final]
sol = solve_ivp(fun, tspan, U0, method="RK45")

U = sol.y

#%%
# -- Solution plot at different times --
levelsP = 20
levelsC = 20

fig, axes = plt.subplots(4, 2, sharex="col", sharey="row", figsize=(4,5))

ax1 = axes[0,0]
ax2 = axes[0,1]
ax3 = axes[1,0]
ax4 = axes[1,1]
ax5 = axes[2,0]
ax6 = axes[2,1]
ax7 = axes[3,0]
ax8 = axes[3,1]

ax1.set_aspect("equal", "box")
ax2.set_aspect("equal", "box")
ax3.set_aspect("equal", "box")
ax4.set_aspect("equal", "box")
ax5.set_aspect("equal", "box")
ax6.set_aspect("equal", "box")
ax7.set_aspect("equal", "box")
ax8.set_aspect("equal", "box")

Nt = sol.t.shape[0]

ax1.tricontourf(coords[:,0], coords[:,1], U[:N,0], cmap=mapa_de_color, levels=levelsP)
ax1.set_title(r"$\Psi$ at $t=%1.3f" %sol.t[0] + "$")

ax2.tricontourf(coords[:,0], coords[:,1], U[N:,0], cmap=mapa_de_color, levels=levelsC)
ax2.set_title(r"$C$ at $t=%1.3f" %sol.t[0] + "$")

ax3.tricontourf(coords[:,0], coords[:,1], U[:N,Nt//3], cmap=mapa_de_color, levels=levelsP)
ax3.set_title(r"$\Psi$ at $t=%1.3f" %sol.t[Nt//3] + "$")

ax4.tricontourf(coords[:,0], coords[:,1], U[N:,Nt//3], cmap=mapa_de_color, levels=levelsC)
ax4.set_title(r"$C$ at $t=%1.3f" %sol.t[Nt//3] + "$")

ax5.tricontourf(coords[:,0], coords[:,1], U[:N,Nt*2//3], cmap=mapa_de_color, levels=levelsP)
ax5.set_title(r"$\Psi$ at $t=%1.3f" %sol.t[Nt*2//3] + "$")

ax6.tricontourf(coords[:,0], coords[:,1], U[N:,Nt*2//3], cmap=mapa_de_color, levels=levelsC)
ax6.set_title(r"$C$ at $t=%1.3f" %sol.t[Nt*2//3] + "$")

ax7.tricontourf(coords[:,0], coords[:,1], U[:N,-1], cmap=mapa_de_color, levels=levelsP)
ax7.set_title(r"$\Psi$ at $t=%1.3f" %sol.t[-1] + "$")

ax8.tricontourf(coords[:,0], coords[:,1], U[N:,-1], cmap=mapa_de_color, levels=levelsC)
ax8.set_title(r"$C$ at $t=%1.3f" %sol.t[-1] + "$")

fig.suptitle(r"Solution with $N=%d" %coords.shape[0] +"$")

# optional: save solution figure
# plt.savefig("Examples/figures/ex6Henry.png", dpi=300, bbox_inches="tight")

plt.show()