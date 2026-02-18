import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-talk"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

from scipy.integrate import solve_ivp

g = cfg.Geometry()

# points
g.point([0,0])                  # 0
g.point([90,0])                 # 1
g.point([90,30])                # 2
g.point([60,30], el_size=0.4)                # 3
g.point([60,16], el_size=0.1)   # 4
g.point([59,16], el_size=0.1)   # 5
g.point([59,27], el_size=0.6)   # 6
g.point([31,27], el_size=0.6)   # 7
g.point([31,12], el_size=0.1)   # 8
g.point([30,12], el_size=0.1)   # 9
g.point([30,30], el_size=0.4)                # 10
g.point([0,30])                 # 11

# lines
left = 10
right = 11
neumann = 12
g.spline([0,1], marker=neumann)
g.spline([1,2], marker=neumann)
g.spline([2,3], marker=right)
g.spline([3,4], marker=neumann)
g.spline([4,5], marker=neumann)
g.spline([5,6], marker=neumann)
g.spline([6,7], marker=neumann)
g.spline([7,8], marker=neumann)
g.spline([8,9], marker=neumann)
g.spline([9,10], marker=neumann)
g.spline([10,11], marker=left)
g.spline([11,0], marker=neumann)

# surfaces
g.surface([0,1,2,3,4,5,6,7,8,9,10,11])

# geometry plot
plt.figure(figsize=(8,3))
cfv.draw_geometry(g,draw_axis=True)
plt.title("Geometry")


# mesh generation
mesh = cfm.GmshMesh(g,el_size_factor=2)

coords, edof, dofs, bdofs, elementmarkers = mesh.create()
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)

plt.figure(figsize=(8,3))
cfv.draw_mesh(
    coords=coords,
    edof=edof,
    dofs_per_node=mesh.dofs_per_node,
    el_type=mesh.el_type,
    filled=True
)
plt.title(f"Mesh")
plt.suptitle(f"el_size_factor={mesh.el_size_factor}, N={coords.shape[0]} nodes", fontsize=8, y=0.90)


# nodes identification
left_nodes = np.asarray(bdofs[left]) - 1
right_nodes = np.asarray(bdofs[right]) - 1
neumann_nodes = np.asarray(bdofs[neumann]) - 1

# elination of repited nodes
neumann_nodes = np.setdiff1d(neumann_nodes, right_nodes)
neumann_nodes = np.setdiff1d(neumann_nodes, left_nodes)

N = coords.shape[0]
boundary_nodes = np.hstack((left_nodes, right_nodes, neumann_nodes))
interior_nodes = np.setdiff1d(np.arange(N), boundary_nodes)

# plot nodes
nodes_to_plot = (
    interior_nodes,
    left_nodes,
    right_nodes,
    neumann_nodes
)
labels = (
    "interior",
    "left",
    "right",
    "neumann",
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
plt.axis("equal")
plt.legend(loc="center")





# Problem Discretization
# Paramters laplacian
L = np.array([0,0,0,1,0,1])
source = lambda p: 0
k = lambda p: 0.5
neumann_condition = lambda p: 0

from GFDMI import GFDMI_2D_problem as gfdmi

problem = gfdmi(coords, faces, L, source)

problem.add_material("interior", k, interior_nodes)
problem.add_dirichlet_boundary("left", left_nodes, lambda p: 50)
problem.add_dirichlet_boundary("right", right_nodes, lambda p: 35)
problem.add_neumann_boundary("neumann", k, neumann_nodes, neumann_condition)

K, F = problem.create_system_K_F()

import scipy.sparse as sp
U = sp.linalg.spsolve(K, F)

# Plot solution
plt.figure(figsize=(8,3))
plt.tricontourf(
    coords[:,0],
    coords[:,1],
    U,
    levels=25,
    cmap="plasma"
)
plt.axis("equal")
plt.colorbar(label="total head")
plt.title("Steady State Solution")
plt.tricontour(
    coords[:,0],
    coords[:,1],
    U,
    triangles=faces,
    levels=25,
    colors="k",
    linewidths=1,
    alpha=0.5
)

solid_nodes = np.array([3,4,5,6,7,8,9,10])
plt.fill(
    coords[solid_nodes,0],
    coords[solid_nodes,1],
    color="gray"
)

# draging dam
xs = np.array([
    30.        , 31.57894737, 33.15789474, 34.73684211, 36.31578947,
    37.89473684, 39.47368421, 41.05263158, 42.63157895, 44.21052632,
    45.78947368, 47.36842105, 48.94736842, 50.52631579, 52.10526316,
    53.68421053, 55.26315789, 56.84210526, 58.42105263, 60.        , 30.
])
ys = np.array([
    45.        , 44.77443609, 44.54887218, 44.32330827, 44.09774436,
    43.27302632, 41.99013158, 40.70723684, 39.42434211, 38.14144737,
    37.34817814, 37.04453441, 36.74089069, 36.43724696, 36.13360324,
    35.82995951, 35.52631579, 35.22267206, 33.94736842, 30.        , 30.
])

plt.fill(
    xs,
    ys,
    color="gray"
)

plt.savefig("Examples/figures/ex8.png", dpi=300, bbox_inches="tight")

plt.show()