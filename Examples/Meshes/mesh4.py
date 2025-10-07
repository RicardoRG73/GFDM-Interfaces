show_figures = False
save_mesh_to_file = True

#%%
# =============================================================================
# Libraries
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind", "seaborn-v0_8-talk"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
plt.rcParams["legend.framealpha"] = 0.1

import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

#%%
# =============================================================================
# Geometry
# =============================================================================
g = cfg.Geometry()

# points
omega_length = 4
omega_height = 1
g.point([0, 0])      # 0
g.point([omega_length, 0])      # 1
g.point([omega_length, omega_height])      # 2
g.point([0, omega_height])      # 3

# lines
left = 10
right = 11
top = 12
bottom = 15

g.line([0,1], marker=bottom, el_on_curve=40)    # 0
g.line([1,2], marker=right, el_on_curve=10)     # 1
g.line([2,3], marker=top, el_on_curve=40)       # 2
g.line([3,0], marker=left, el_on_curve=10)      # 3


# surfaces
mat0 = 0
g.struct_surface([0,1,2,3], marker=mat0)

#%%
# =============================================================================
# Mesh
# =============================================================================
mesh = cfm.GmshMesh(g,el_size_factor=0.4)

coords, edof, dofs, bdofs, elementmarkers = mesh.create()
verts, faces, vertices_per_face, is_3d = cfv.ce2vf(
    coords,
    edof,
    mesh.dofs_per_node,
    mesh.el_type
)



#%%
# =============================================================================
# Nodes identification by color
# =============================================================================
corner_nodes = np.array([0,1,2,3])

left_nodes = np.asarray(bdofs[left]) - 1
left_nodes = np.setdiff1d(left_nodes, corner_nodes)
left_nodes = np.hstack((left_nodes, [0,3]))

right_nodes = np.asarray(bdofs[right]) - 1
right_nodes = np.setdiff1d(right_nodes, corner_nodes)
right_nodes = np.hstack((right_nodes, [1,2]))

bottom_nodes = np.asarray(bdofs[bottom]) - 1
bottom_nodes = np.setdiff1d(bottom_nodes, corner_nodes)

top_nodes = np.asarray(bdofs[top]) - 1
top_nodes = np.setdiff1d(top_nodes, corner_nodes)

boundaries = np.hstack((
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_nodes
))

N = coords.shape[0]
interior_nodes = np.setdiff1d(np.arange(N), boundaries)

nodes_to_plot = (
    interior_nodes,
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_nodes
)
labels = (
    "Interior",
    "Left",
    "Right",
    "Bottom",
    "Top"
)

boundaries = np.hstack((
    left_nodes,
    right_nodes,
    bottom_nodes,
    top_nodes
))

if show_figures:
    # plotting geometry
    cfv.figure()
    cfv.title('Geometry')
    cfv.draw_geometry(g, draw_axis=True)
    # plt.savefig("figures/05bgeometry.jpg", dpi=300)

    # plotting mesh
    cfv.figure()
    cfv.title('Malla $N=%d' %coords.shape[0] +'$')
    cfv.draw_mesh(
        coords=coords,
        edof=edof,
        dofs_per_node=mesh.dofs_per_node,
        el_type=mesh.el_type,
        filled=True
    )
    # plt.savefig("figures/05bmesh.jpg", dpi=300)

    # plotting nodes by color
    plt.figure()
    for nodes,label in zip(nodes_to_plot, labels):
        plt.scatter(
            coords[nodes,0],
            coords[nodes,1],
            label=label,
            alpha=0.75
        )
    plt.axis("equal")
    plt.legend()
    # plt.savefig("figures/05bnodes.jpg", dpi=300)


    plt.show()



if save_mesh_to_file:
    import json
    data_to_save = {}
    for b,label in zip(nodes_to_plot, labels):
        data_to_save[label.replace(" ","_").replace("-","_").lower()+"_nodes"] = b.tolist()
    data_to_save["coords"] = coords.tolist()
    data_to_save["triangles"] = faces.tolist()
    data_to_save["boundaries"] = boundaries.tolist()
    data_to_save["omega_length"] = omega_length
    data_to_save["omega_height"] = omega_height

    with open('Examples/Meshes/mesh4.json', 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print("\n ============\n Mesh saved \n ============")