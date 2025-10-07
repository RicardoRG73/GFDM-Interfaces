show_plots = True
save_mesh_to_file = False

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

# calfem-python
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

#%%
# =============================================================================
# Creating geometry object
# =============================================================================
geometry = cfg.Geometry()

# points: square domain
geometry.point([0,0])       # 0
geometry.point([1,0])       # 1
geometry.point([1,1])       # 2
geometry.point([0,1])       # 3

# points: interface-boundaries intersection
delta_interface = 0.01

geometry.point([ 0.5 - delta_interface , 0 ])     # 4
geometry.point([ 0.5 - delta_interface , 1 ])     # 5
geometry.point([ 0.5 + delta_interface , 0 ])     # 6
geometry.point([ 0.5 + delta_interface , 1 ])     # 7

# lines: square domain
dirichlet = 10
geometry.spline([5,3], marker=dirichlet)    # 0
geometry.spline([3,0], marker=dirichlet)    # 1
geometry.spline([0,4], marker=dirichlet)    # 2

geometry.spline([6,1], marker=dirichlet)    # 3
geometry.spline([1,2], marker=dirichlet)    # 4
geometry.spline([2,7], marker=dirichlet)    # 5

# interface
## left interface
interface_left = 11
### points
N = 11
delta_y = 1/(N+1)
y = 0

for i in range(N):
    y += delta_y
    x = - delta_interface + 0.5 + 0.1 * np.sin(6.28 * y)
    geometry.point([x, y])

### lines
for i in range(N-1):
    geometry.spline([8+i,9+i], marker=interface_left)

### left interface lines, conecting interface and boundaries
geometry.spline([4,8], marker=interface_left)
geometry.spline([7+N,5], marker=interface_left)

## right interface
interface_right = 12
### points
y = 0
for i in range(N):
    y += delta_y
    x = delta_interface + 0.5 + 0.1 * np.sin(6.28 * y)
    geometry.point([x, y])

### lines
for i in range(N-1):
    geometry.spline([8+N+i,9+N+i], marker=interface_right)

### left interface lines, conecting interface and boundaries
geometry.spline([6,8+N], marker=interface_right)
geometry.spline([7+2*N,7], marker=interface_right)


# surfaces
## \Omega^+ : left side
left_domain = 0
left_surf_index = np.hstack((
    np.array([0,1,2]),
    np.array([5+N]),
    np.arange(5+1,5+N),
    np.array([5+N+1])
))
geometry.surface(left_surf_index, marker=left_domain)

## \Omega^- : right side
right_domain = 1
left_surf_index = np.hstack((
    np.array([5+2*N+1]),
    np.arange(5+N+2,5+2*N+1),
    np.array([5+2*N+2]),
    np.array([5,4,3])
))
geometry.surface(left_surf_index, marker=right_domain)

#%%
# =============================================================================
# Creating mesh
# =============================================================================
mesh = cfm.GmshMesh(geometry)

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

#%%
# =============================================================================
# Nodes indexing separated by boundary conditions
# =============================================================================
# Dirichlet nodes
dirichlet_nodes = np.asarray(bdofs[dirichlet]) - 1

# Interface nodes
left_interface_nodes = np.asarray(bdofs[interface_left]) - 1
left_interface_nodes = np.setdiff1d(left_interface_nodes, [4,5])
right_interface_nodes = np.asarray(bdofs[interface_right]) - 1
right_interface_nodes = np.setdiff1d(right_interface_nodes, [6,7])

# Interior nodes
elementmarkers = np.asarray(elementmarkers)
boundaries = np.hstack((dirichlet_nodes,left_interface_nodes,right_interface_nodes))

left_interior_nodes = faces[elementmarkers == left_domain]
left_interior_nodes = left_interior_nodes.flatten()
left_interior_nodes = np.setdiff1d(left_interior_nodes,boundaries)

right_interior_nodes = faces[elementmarkers == right_domain]
right_interior_nodes = right_interior_nodes.flatten()
right_interior_nodes = np.setdiff1d(right_interior_nodes,boundaries)

nodes = (
    dirichlet_nodes,
    left_interface_nodes,
    right_interface_nodes,
    left_interior_nodes,
    right_interior_nodes
)
labels=(
    "Dirichlet",
    "Interface left",
    "Interface right",
    "Omega_plus",
    "Omega_minus"
)

if save_mesh_to_file:
    import json
    data_to_save = {}
    for b,label in zip(nodes, labels):
        data_to_save[label.replace(" ","_").replace("-","_").lower()+"_nodes"] = b.tolist()
    data_to_save["coords"] = coords.tolist()
    data_to_save["triangles"] = faces.tolist()
    with open('Examples/Meshes/mesh2.json', 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print("\n ============\n Mesh saved \n ============")


if show_plots:
    # geometry plot
    cfv.figure(fig_size=(4,4))
    cfv.title('Geometry')
    cfv.draw_geometry(geometry)
    # plt.savefig("figures/03geometry.jpg", dpi=300)

    # mesh plot
    cfv.figure(fig_size=(8,4))
    cfv.title('Mesh')
    cfv.draw_mesh(coords=coords, edof=edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, filled=True)
    # plt.savefig("figures/03mesh.jpg", dpi=300)

    # ploting boundaries in different colors
    plt.figure()
    for b,label in zip(nodes, labels):
        plt.scatter(coords[b,0], coords[b,1], label=label)
    plt.axis("equal")
    plt.title("$N = %d$" %coords.shape[0])
    plt.legend()
    # plt.savefig("figures/03nodes.jpg", dpi=300)


    plt.show()