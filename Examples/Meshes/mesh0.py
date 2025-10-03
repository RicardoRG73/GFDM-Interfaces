save_mesh_to_file = False
show_plots = True

#%% Importing Needed libraries
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

#%% Creating geometry object
geometry = cfg.Geometry()                   # geometry object

# points
geometry.point([0,0])                      # 0
geometry.point([1,0])                       # 1
geometry.point([2,0])                       # 2
geometry.point([1,1])                       # 3
geometry.point([0,1])                      # 4

# lines
left = 11                                   # marker for nodes on left boundary
right = 12                                  # marker for nodes on right boundary
bottom = 13                                 # marker for bottom nodes
top = 14                                    # marker for top nodes
geometry.spline([0,1], marker=bottom)       # 0
geometry.spline([1,2], marker=bottom)       # 1
geometry.circle([2,1,3], marker=right)      # 2
geometry.spline([3,4], marker=top)          # 3
geometry.spline([4,0], marker=left)         # 4


# surfaces
mat0 = 100                                  # marker for nodes on material 1
geometry.surface([0,1,2,3,4], marker=mat0)  # 0

#%% Creating mesh from geometry object
mesh = cfm.GmshMesh(geometry)

mesh.el_type = 2                            # type of element: 2 = triangle
mesh.dofs_per_node = 1
mesh.el_size_factor = 0.08

coords, edof, dofs, bdofs, elementmarkers = mesh.create()   # create the geometry
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)


#%% Nodes indexing separated by boundary conditions
left_nodes = np.asarray(bdofs[left]) - 1                # index of nodes on left boundary
right_nodes = np.asarray(bdofs[right]) - 1               # index of nodes on right boundary
right_nodes = np.setdiff1d(right_nodes, [2,3])
bottom_nodes = np.asarray(bdofs[bottom]) - 1
bottom_nodes = np.setdiff1d(bottom_nodes, [0])
top_nodes = np.asarray(bdofs[top]) - 1
top_nodes = np.setdiff1d(top_nodes, [4])

B = np.hstack((
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_nodes
))                          # all boundaries

elementmarkers = np.asarray(elementmarkers)

interior_nodes = faces[elementmarkers == mat0]
interior_nodes = interior_nodes.flatten()
interior_nodes = np.setdiff1d(interior_nodes,B)

nodes = (left_nodes,right_nodes,bottom_nodes,top_nodes,interior_nodes)
labels = ("Left", "Right", "Bottom", "Top", "Interior")

if save_mesh_to_file:
    import json
    data_to_save = {}
    for b,label in zip(nodes, labels):
        data_to_save[label.lower()+"_nodes"] = b.tolist()
    data_to_save["coords"] = coords.tolist()
    data_to_save["triangles"] = faces.tolist()
    with open('Examples/Meshes/mesh0.json', 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print("\n ============\n Mesh saved \n ============")


if show_plots:
    # geometry plot
    cfv.figure()
    cfv.title('Geometry')
    cfv.draw_geometry(geometry)
    # plt.savefig("figures/00geometry.jpg", dpi=300)

    # mesh plot
    cfv.figure(fig_size=(8,4))
    cfv.title('Mesh')
    cfv.draw_mesh(coords=coords, edof=edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, filled=True)
    # plt.savefig("figures/00mesh.jpg", dpi=300)

    # ploting boundaries in different color
    plt.figure()
    for b,label in zip(nodes, labels):
        plt.scatter(coords[b,0], coords[b,1], label=label)
    plt.axis("equal")
    plt.title("$N = %d$" %coords.shape[0])
    plt.legend(loc="center")
    # plt.savefig("figures/00nodes.jpg", dpi=300)

    plt.show()