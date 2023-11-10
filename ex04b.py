# =============================================================================
# Importing needed libraries
# =============================================================================
import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

import numpy as np
import matplotlib.pyplot as plt

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind"])#, "seaeborn-v0_8-paper"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
cmap_color = "plasma"

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

Hizq=108
Hdere=100
Hnbottom=101
Hntop=102

InterfaceA = 110
InterfaceB = 111

g.spline([0, 1], marker=Hnbottom) #0
g.spline([1, 2], marker=Hnbottom) #1
g.spline([2, 3], marker=Hnbottom) #2
g.spline([3, 4], marker=Hdere,el_on_curve=15) #3
g.spline([4, 5], marker=Hntop) #4
g.spline([5, 6], marker=Hntop) #5
g.spline([6, 7], marker=Hntop, el_on_curve=10) #6
g.spline([7, 0], marker=Hizq,el_on_curve=15) #7
g.spline([1, 6], marker=InterfaceA) #8
g.spline([2, 5], marker=InterfaceB) #9

Rockfill=100
Clay=102
 
g.surface([0, 8, 6, 7],marker=Rockfill) #0
g.surface([1, 9, 5, 8],marker=Clay) #1
g.surface([2,3,4,9],marker=Rockfill) #2

cfv.figure(fig_size=(6,4))
cfv.title('Geometry')
cfv.draw_geometry(g)

# =============================================================================
# Mesh
# =============================================================================
mesh = cfm.GmshMesh(g)

mesh.el_type = 2  #2= triangulo de 3 nodos 9= triangulo de 6 nodos
mesh.dofs_per_node = 1  # Degrees of freedom per node.
mesh.el_size_factor = 2.0  # Factor that changes element sizes.

coords, edof, dofs, bdofs, element_markers = mesh.create()

nnel=len(edof[1,:])
Elementos=np.zeros(edof.shape,dtype=int)
for i,elem in enumerate(edof):
        if nnel==3:
            Elementos[i,:]=elem[1],elem[0],elem[2]
        elif nnel==6:
            Elementos[i,:]=elem[1],elem[3],elem[0],elem[5],elem[2],elem[4]
Elementos=Elementos-1  
bdofs={frontera:np.array(bdofs[frontera])-1 for frontera in bdofs}

cfv.figure(fig_size=(6,4))
cfv.title('Mesh')
cfv.draw_mesh(coords=coords, edof=edof, dofs_per_node=mesh.dofs_per_node, el_type=mesh.el_type, filled=True)

# =============================================================================
# Nodes index
# =============================================================================
bizq = np.asarray(bdofs[Hizq])
bder = np.asarray(bdofs[Hdere])
bnb = np.asarray(bdofs[Hnbottom])
bnb = np.setdiff1d(bnb , np.intersect1d(bnb,bizq))
bnb = np.setdiff1d(bnb , np.intersect1d(bnb,bder))
bnt = np.asarray(bdofs[Hntop])
bnt = np.setdiff1d(bnt , np.intersect1d(bnt,bizq))
bnt = np.setdiff1d(bnt , np.intersect1d(bnt,bder))
bia = np.asarray(bdofs[InterfaceA])
bia = np.setdiff1d(bia, np.intersect1d(bia, bnt))
bia = np.setdiff1d(bia, np.intersect1d(bia, bnb))
bib = np.asarray(bdofs[InterfaceB])
bib = np.setdiff1d(bib, np.intersect1d(bib, bnt))
bib = np.setdiff1d(bib, np.intersect1d(bib, bnb))

B = np.hstack((bizq, bder, bnb, bnt, bia, bib))

element_markers = np.array(element_markers)

rock = Elementos[element_markers == Rockfill]
rock = rock.flatten()
rock = np.setdiff1d(rock, B)

clay = Elementos[element_markers == Clay]
clay = clay.flatten()
clay = np.setdiff1d(clay, B)

# Plot nodes by color
plt.figure(figsize=(6,2))
opacity = 0.5
node_size = 40

plt.scatter(coords[bizq,0], coords[bizq,1], label="bizq", alpha=opacity)
plt.scatter(coords[bder,0], coords[bder,1], label="bder", alpha=opacity)
plt.scatter(coords[bnb,0], coords[bnb,1], label="bnb", alpha=opacity)
plt.scatter(coords[bnt,0], coords[bnt,1], label="bnt", alpha=opacity)
plt.scatter(coords[bia,0], coords[bia,1], label="bia", alpha=opacity)
plt.scatter(coords[bib,0], coords[bib,1], label="bib", alpha=opacity)

plt.scatter(coords[rock,0], coords[rock,1], label="rock", alpha=opacity)
plt.scatter(coords[clay,0], coords[clay,1], label="clay", alpha=opacity)

plt.axis("equal")
plt.legend()

# =============================================================================
# Problem parameters
# =============================================================================
