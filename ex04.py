import calfem.geometry as cfg
import calfem.mesh as cfm
import calfem.vis_mpl as cfv

import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp

plt.style.use(["seaborn-v0_8-darkgrid", "seaborn-v0_8-colorblind"])#, "seaeborn-v0_8-paper"])
plt.rcParams["legend.frameon"] = True
plt.rcParams["legend.shadow"] = True
cmap_color = "plasma"

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


g.spline([0, 1], marker=Hnbottom) #0
g.spline([1, 2], marker=Hnbottom) #1
g.spline([2, 3], marker=Hnbottom) #2
g.spline([3, 4], marker=Hdere,el_on_curve=15) #3
g.spline([4, 5], marker=Hntop) #4
g.spline([5, 6], marker=Hntop) #5
g.spline([6, 7], marker=Hntop, el_on_curve=10) #6
g.spline([7, 0], marker=Hizq,el_on_curve=15) #7
g.spline([1, 6]) #8
g.spline([2, 5]) #9

Rockfill=100
Clay=102
 
g.surface([0, 8, 6, 7],marker=Rockfill) #0
g.surface([1, 9, 5, 8],marker=Clay) #1
g.surface([2,3,4,9],marker=Rockfill) #2

cfv.figure(fig_size=(6,4))
cfv.title('Geometry')
cfv.draw_geometry(g)


mesh = cfm.GmshMesh(g)

mesh.el_type = 2  #2= triangulo de 3 nodos 9= triangulo de 6 nodos
mesh.dofs_per_node = 1  # Degrees of freedom per node.
mesh.el_size_factor = 2.0  # Factor that changes element sizes.

coords, edof, dofs, bdofs, element_markers = mesh.create()

#Acondiciomamiento de la malla
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


bizq = np.asarray(bdofs[Hizq])
bder = np.asarray(bdofs[Hdere])
bnb = np.asarray(bdofs[Hnbottom])
bnb = np.setdiff1d(bnb , np.intersect1d(bnb,bizq))
bnb = np.setdiff1d(bnb , np.intersect1d(bnb,bder))
bnt = np.asarray(bdofs[Hntop])
bnt = np.setdiff1d(bnt , np.intersect1d(bnt,bizq))
bnt = np.setdiff1d(bnt , np.intersect1d(bnt,bder))


B = np.hstack((bizq, bder, bnb, bnt))

interior = np.setdiff1d(
    np.arange(coords.shape[0]),
    B
)

plt.figure(figsize=(6,2))
opacity = 0.5
node_size = 40

# plt.scatter(coords[interior,0], coords[interior,1], label="interior", alpha=opacity)

plt.scatter(coords[bizq,0], coords[bizq,1], label="bizq", alpha=opacity)
plt.scatter(coords[bder,0], coords[bder,1], label="bder", alpha=opacity)
plt.scatter(coords[bnb,0], coords[bnb,1], label="bnb", alpha=opacity)
plt.scatter(coords[bnt,0], coords[bnt,1], label="bnt4", alpha=opacity)

plt.axis("equal")
plt.legend()


L = np.array([0,0,0,1,0,1])
k = lambda p: 1
source = lambda p: 0
neu_cond = lambda p: 0
dir_izq = lambda p: 8
dir_der = lambda p: 0


from GFDMI import create_system_K_F

material = {}
material["0"] = [k, interior]

neumann = {}
neumann["0"] = [k, bnb, neu_cond]
neumann["1"] = [k, bnt, neu_cond]

dirichlet = {}
dirichlet["izq"] = [bizq, dir_izq]
dirichlet["der"] = [bder, dir_der]

interfaces = {}

K,F = create_system_K_F(
    p=coords,
    triangles=Elementos,
    L=L,
    source=source,
    materials=material,
    neumann_boundaries=neumann,
    dirichlet_boundaries=dirichlet,
    interfaces = interfaces
)

U = sp.linalg.spsolve(K,F)

# =============================================================================
# Plots
# =============================================================================

# 3D
plt.figure()
ax = plt.axes(projection="3d")
ax.plot_trisurf(
    coords[:,0],
    coords[:,1],
    U,
    cmap=cmap_color
)

# Contourf
plt.figure()
plt.tricontourf(
    coords[:,0],
    coords[:,1],
    U,
    cmap=cmap_color,
    levels=20
)
plt.axis("equal")
plt.colorbar()
# line h=0
plt.tricontour(
    coords[:,0],
    coords[:,1],
    (U - coords[:,1])*9.81,
    levels=[0.0],
    colors="b"
)

# =============================================================================
# No estacionario (Ecuación de Difusión)
# \nabla^2 u + f = du/dt
# =============================================================================
from scipy.integrate import solve_ivp

F = F.toarray().flatten()


t = [0,200]
fun = lambda t,U: K@U - F
U0 = np.zeros(coords.shape[0])
U0[bizq] = 8
U0[bder] = 0
# U0[bnb] = 8*(1 - coords[bnb,0]/40)
# U0[bnt] = coords[bnt,1]

plt.figure()
ax = plt.axes(projection="3d")
ax.plot_trisurf(
    coords[:,0],
    coords[:,1],
    U0,
    cmap=cmap_color
)
ax.set_title("Condición inicial $U_0$")

sol = solve_ivp(fun, t, U0)

U2 = sol.y
# plots
cmap_color = "summer"
fig = plt.figure()
# 1
indice = 0
ax = plt.subplot(2,2,1)
ax.tricontourf(
    coords[:,0],
    coords[:,1],
    U2[:,indice],
    cmap=cmap_color,
    levels=20
)
ax.tricontour(
    coords[:,0],
    coords[:,1],
    (U2[:,indice] - coords[:,1])*9.81,
    levels=[0.0],
    colors="b"
)
ax.axis("equal")
ax.set_title("$t = %1.4f$" %sol.t[indice])
# 2
indice = U2.shape[1] // 3
ax = plt.subplot(2,2,2)
ax.tricontourf(
    coords[:,0],
    coords[:,1],
    U2[:,indice],
    cmap=cmap_color,
    levels=20
)
ax.tricontour(
    coords[:,0],
    coords[:,1],
    (U2[:,indice] - coords[:,1])*9.81,
    levels=[0.0],
    colors="b"
)
ax.axis("equal")
ax.set_title("$t = %1.4f$" %sol.t[indice])
# 3
indice = U2.shape[1] // 3 * 2
ax = plt.subplot(2,2,3)
ax.tricontourf(
    coords[:,0],
    coords[:,1],
    U2[:,indice],
    cmap=cmap_color,
    levels=20
)
ax.tricontour(
    coords[:,0],
    coords[:,1],
    (U2[:,indice] - coords[:,1])*9.81,
    levels=[0.0],
    colors="b"
)
ax.axis("equal")
ax.set_title("$t = %1.4f$" %sol.t[indice])
# 4
indice = -1
ax = plt.subplot(2,2,4)
ax.tricontourf(
    coords[:,0],
    coords[:,1],
    U2[:,indice],
    cmap=cmap_color,
    levels=20
)
ax.tricontour(
    coords[:,0],
    coords[:,1],
    (U2[:,indice] - coords[:,1])*9.81,
    levels=[0.0],
    colors="b"
)
ax.axis("equal")
ax.set_title("$t = %1.4f$" %sol.t[indice])


plt.figure()
ax = plt.axes(projection="3d")
ax.plot_trisurf(
    coords[:,0],
    coords[:,1],
    U2[:,-1],
    cmap="plasma"
)
ax.set_title("Sol $U$ en $t=%1.4f$" %sol.t[-1])

print("\n\nNúmero de condición de K: %1.3e" %np.linalg.cond(K.toarray()))

plt.show()