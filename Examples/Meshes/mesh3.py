
show_plots = True
save_mesh_to_file = True
#%%
# =============================================================================
# Importing needed libraries
# =============================================================================
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv
import matplotlib.pyplot as plt
import numpy as np
plt.style.use("seaborn-v0_8")

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

if show_plots:
    # geometry plot
    cfv.figure(fig_size=(6,4))
    cfv.title('Geometry')
    cfv.draw_geometry(g)
    # plt.savefig("figures/04bgeometry.jpg", dpi=300)

    # mesh plot
    cfv.figure(fig_size=(6,4))
    cfv.title('Mesh')
    cfv.draw_mesh(coords=coords, edof=edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, filled=True)
    # plt.savefig("figures/04bmesh.jpg", dpi=300)

    # plotting nodes by color
    plt.figure()
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
    # plt.savefig("figures/04bnodes.jpg", dpi=300)

    plt.show()

if save_mesh_to_file:
    import json
    data_to_save = {}
    for b,label in zip(nodes_to_plot, labels):
        data_to_save[label.replace(" ","_").replace("-","_").lower()+"_nodes"] = b.tolist()
    data_to_save["coords"] = coords.tolist()
    data_to_save["triangles"] = triangles.tolist()
    with open('Examples/Meshes/mesh3.json', 'w') as file:
        json.dump(data_to_save, file, indent=4)
    print("\n ============\n Mesh saved \n ============")