save_mesh_to_file = False
show_plots = True

#%% Importing needed libraries
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8")
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["figure.autolayout"] = True

# calfem-python
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv


#%% Creating geometry
geometry = cfg.Geometry()                       # geometry object

# points
geometry.point([-1,0])                          # 0
geometry.point([1,0])                           # 1
geometry.point([1,1])                           # 2
geometry.point([-1,1])                          # 3

delta = 0.0005   
interface_offset = 0.2
el_size_scale_factor = 1
geometry.point([-interface_offset-delta, 0], el_size=el_size_scale_factor)    # 4
geometry.point([-interface_offset+delta, 0], el_size=el_size_scale_factor)    # 5
geometry.point([interface_offset-delta, 1], el_size=el_size_scale_factor)     # 6
geometry.point([interface_offset+delta, 1], el_size=el_size_scale_factor)     # 7

# lines
left = 100
neumann_bottom_left = 101
left_interface = 102
neumann_top_left = 103
geometry.spline([3,0], marker=left)             # 0
geometry.spline([0,4], marker=neumann_bottom_left)         # 1
geometry.spline([4,6], marker=left_interface)       # 2
geometry.spline([6,3], marker=neumann_top_left)         # 3

right = 104
neumann_top_right = 105
right_interface = 106
neumann_bottom_right = 107
geometry.spline([1,2], marker=right)            # 4
geometry.spline([2,7], marker=neumann_top_right)         # 5
geometry.spline([5,7], marker=right_interface)       # 6
geometry.spline([5,1], marker=neumann_bottom_right)         # 7

# surfaces
mat0 = 10
mat1 = 11
geometry.surface([0,1,2,3], marker=mat0)        # 0
geometry.surface([4,5,6,7], marker=mat1)        # 1

#%% Creating mesh
mesh = cfm.GmshMesh(geometry)

mesh.el_type = 2                            # type of element: 2 = triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.04

coords, edof, dofs, bdofs, elementmarkers = mesh.create()   # create the geometry
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)

#%% Nodes indexing separated by boundary conditions
# Dirichlet nodes
left_nodes = np.asarray(bdofs[left]) - 1                # index of nodes on left boundary
right_nodes = np.asarray(bdofs[right]) - 1               # index of nodes on right boundary

# Neumann nodes
bottom_left_nodes = np.asarray(bdofs[neumann_bottom_left]) - 1
bottom_left_nodes = np.setdiff1d(bottom_left_nodes, 0)
top_left_nodes = np.asarray(bdofs[neumann_top_left]) - 1
top_left_nodes = np.setdiff1d(top_left_nodes, 3)
top_right_nodes = np.asarray(bdofs[neumann_top_right]) - 1
top_right_nodes = np.setdiff1d(top_right_nodes, 2)
bottom_right_nodes = np.asarray(bdofs[neumann_bottom_right]) - 1
bottom_right_nodes = np.setdiff1d(bottom_right_nodes, 1)

# Interface nodes
left_interface_nodes = np.asarray(bdofs[left_interface]) - 1
left_interface_nodes = np.setdiff1d(left_interface_nodes, [4,6])
right_interface_nodes = np.asarray(bdofs[right_interface]) - 1
right_interface_nodes = np.setdiff1d(right_interface_nodes, [5,7])

# Interior nodes
elementmarkers = np.asarray(elementmarkers)
boundaries = np.hstack((
    left_nodes,
    right_nodes,
    bottom_left_nodes,
    top_left_nodes,
    top_right_nodes,
    bottom_right_nodes,
    left_interface_nodes,
    right_interface_nodes
))

interior_mat0 = faces[elementmarkers == mat0]
interior_mat0 = interior_mat0.flatten()
interior_mat0 = np.setdiff1d(interior_mat0,boundaries)

interior_nodes_mat1 = faces[elementmarkers == mat1]
interior_nodes_mat1 = interior_nodes_mat1.flatten()
interior_nodes_mat1 = np.setdiff1d(interior_nodes_mat1,boundaries)


nodes = (
    left_nodes,
    right_nodes,
    bottom_left_nodes,
    top_left_nodes,
    top_right_nodes,
    bottom_right_nodes,
    left_interface_nodes,
    right_interface_nodes,
    interior_mat0,
    interior_nodes_mat1
)
labels = (
    "Left",
    "Right",
    "Bottom-Left",
    "Top-Left",
    "Top-Right",
    "Bottom-Right",
    "Left Interface",
    "Right Interface",
    "Interior Material 0",
    "Interior Material 1"
)


if save_mesh_to_file:
    import json
    data_to_save = {}
    for b,label in zip(nodes, labels):
        data_to_save[label.replace(" ","_").replace("-","_").lower()+"_nodes"] = b.tolist()
    data_to_save["coords"] = coords.tolist()
    data_to_save["triangles"] = faces.tolist()
    with open('Examples/Meshes/mesh1.json', 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print("\n ============\n Mesh saved \n ============")


if show_plots:
    #%% geometry plot
    cfv.figure(fig_size=(8,4))
    cfv.title('Geometry')
    cfv.draw_geometry(geometry)
    # plt.savefig("figures/01geometry.jpg", dpi=300)

    #%% mesh plot
    cfv.figure(fig_size=(8,4))
    cfv.title('Mesh')
    cfv.draw_mesh(coords=coords, edof=edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, filled=True)
    # plt.savefig("figures/01mesh.jpg", dpi=300)

    #%% ploting boundaries in different colors
    plt.figure()
    for b,label in zip(nodes, labels):
        plt.scatter(coords[b,0], coords[b,1], label=label, alpha=0.5, s=20)
    plt.axis("equal")
    plt.title("$N = %d$" %coords.shape[0])
    plt.legend(loc="center")
    # plt.savefig("figures/01nodes.jpg", dpi=300)
    plt.show()