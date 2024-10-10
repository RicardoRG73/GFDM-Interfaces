"""
Generalized Finite Differences Method
Applied to layered materials (Interfaces)
Using General Neumann Boundaries
"""
import numpy as np
import scipy.sparse as sp



def support_nodes(i,triangles, min_support_nodes=5, max_iter=2):
    """
    Returns the index of support nodes `I` correspondig to the central node
    with index `ì`. 
    
    Parameters
    ----------
    i : int
        index of central node.
    triangles : numpy.ndarray
        array with shape (n,3), containing index of the n triangles
        with 3 nodes each.
    min_support_nodes : int, optional
        number of minimum support nodes. The default is 5.
    max_iter : int, optional
        number of maximun iterations for adding support nodes to the list `I`. The default is 2.

    Returns
    -------
    I : numpy.ndarray
        index of the support nodes of central node `i`.
    
    """
    I = np.array([i])                                                   # Initialices I with the index of central node i
    iter = 1                                                            # Initialices interation count
    while I.shape[0] < min_support_nodes and iter <= max_iter:          # Checks support nodes (minimum 5) and iteration (max 2)
        temp =  np.any( np.isin(triangles,I), axis=1)                   # Boolean array for triangles containing center node i
        temp = triangles[temp,:].flatten()                              # Keeping triangles contaning center node i, and flatten as 0 dimension array
        I = np.unique(temp)                                             # Deleting repetitions
        iter += 1                                                       # Increase the iteration counter
    return I



def normal_vectors(b,p):
    """
    Computes normal vectors `n` of boundary nodes `b`, using their
    coordinates `p`.

    Parameters
    ----------
    b : numpy.ndarray
        Index of boundary nodes.
    p : numpy.ndarray
        2D coordinates of all domain nodes, with shape (num_nodes,2).

    Returns
    -------
    n : numpy.ndarray
        Normal vectors for nodes `b`, with shape (len(b),2).
    
    """
    percentage_line_tolerance = 0.99
    N = b.shape[0]
    l1 = p[b[1],:] - p[b[0],:]
    l1 = l1 / np.linalg.norm(l1)
    l2 = p[b[N//2],:] - p[b[0],:]
    l2 = l2 / np.linalg.norm(l2)
    line = np.dot(l1,l2) > percentage_line_tolerance
    rotation = np.array([[0,1],[-1,0]])
    if line:
        l1 = rotation @ l1
        n = np.kron(np.ones(N),l1).reshape((N,2))
    else: # curve
        n = np.zeros((N,2))
        centroid = np.mean(p,axis=0)
        for i in b:
            d = np.sqrt((p[i,0]-p[b,0])**2 + (p[i,1]-p[b,1])**2)
            imin = b[d.argsort()[:7]]
            pm = np.mean(p[imin,:], axis=0)
            v1 = p[imin[5]] - pm
            v2 = p[imin[6]] - pm
            ni = rotation @ (v2-v1) / np.linalg.norm(v2-v1)
            ni = ni * np.dot(ni , p[i]-centroid)
            ni = ni / np.linalg.norm(ni)
            n[b==i] = ni
    
    return n



def deltas_matrix(deltasx, deltasy):
    """
    Creates caracteristic-distances-matrix `M` of GFDM system `M\Gamma=L`. 

    Parameters
    ----------
    deltasx : numpy.ndarray
        Horizontal distances from central node `p_0` to support nodes `p_i`.
    deltasy : numpy.ndarray
        Vertical distances from central node `p_0` to support nodes `p_i`.

    Returns
    -------
    M : numpy.ndarray
        Distances matrix.
    
    """
    M = np.vstack((
        np.ones(deltasx.shape),
        deltasx,
        deltasy,
        deltasx**2,
        deltasx*deltasy,
        deltasy**2
    ))
    return M



def interior_assembling(p,b,triangles,L,k,source,K,F):
    for i in b:
        I = support_nodes(i,triangles)
        deltasx = p[I,0] - p[i,0]
        deltasy = p[I,1] - p[i,1]
        M = deltas_matrix(deltasx, deltasy)
        Gamma = np.linalg.pinv(M) @ L
        K[i,I] = Gamma
        F[i] = source(p[i])
    return K, F



def dirichlet_assembling(p,b,u,K,F):
    for i in b:
        F[i] = u(p[i])
        K[i,i] = 1
    return K, F



def interface_assembling(p,b,triangles,m1,m0,k0,k1,source,L,beta,K,F):
    # material 0
    n = normal_vectors(b,p)
    for i in b:
        I0 = support_nodes(i,triangles)
        I0 = np.setdiff1d(I0,m1)

        ni = n[b==i][0]

        deltasx = p[I0,0] - p[i,0]
        deltasy = p[I0,1] - p[i,1]
        ghost = np.array([-np.mean(deltasx), -np.mean(deltasy)])
        norm_ghost = np.linalg.norm(ghost)
        ghostx, ghosty = norm_ghost * ni

        norm_ghost = np.linalg.norm(np.array([ghostx, ghosty]))
        mean_delta = np.mean(np.sqrt(deltasx**2 + deltasy**2)[1:])
        scale_factor = mean_delta/norm_ghost
        ghostx = scale_factor * ghostx
        ghosty = scale_factor * ghosty

        deltasx = np.hstack((ghostx, deltasx))
        deltasy = np.hstack((ghosty, deltasy))

        M = deltas_matrix(deltasx, deltasy)
        k0_i = k0(p[i])
        Gamma = np.linalg.pinv(M) @ (k0_i*L)
        Gamma_ghost = Gamma[0]
        Gamma = Gamma[1:]

        nx, ny = ni
        Gamma_n = np.linalg.pinv(M) @ (k0_i*np.array([0,nx,ny,0,0,0]))
        Gamma_n_ghost = Gamma_n[0]
        Gamma_n = Gamma_n[1:]
        Gg = Gamma_ghost / Gamma_n_ghost
        beta_i = beta(p[i])
        K[i,I0] = Gamma - Gg * Gamma_n
        F[i] = source(p[i]) - Gg * beta_i
    
    # material 1
    for i in b:
        I1 = support_nodes(i,triangles)
        I1 = np.setdiff1d(I1,m0)

        ni = -n[b==i][0]

        deltasx = p[I1,0] - p[i,0]
        deltasy = p[I1,1] - p[i,1]
        ghost = np.array([-np.mean(deltasx), -np.mean(deltasy)])
        norm_ghost = np.linalg.norm(ghost)
        ghostx, ghosty = norm_ghost * ni

        norm_ghost = np.linalg.norm(np.array([ghostx, ghosty]))
        mean_delta = np.mean(np.sqrt(deltasx**2 + deltasy**2)[1:])
        scale_factor = mean_delta/norm_ghost
        ghostx = scale_factor * ghostx
        ghosty = scale_factor * ghosty

        deltasx = np.hstack((ghostx, deltasx))
        deltasy = np.hstack((ghosty, deltasy))

        M = deltas_matrix(deltasx, deltasy)
        k1_i = k1(p[i])
        Gamma = np.linalg.pinv(M) @ (k1_i*L)
        Gamma_ghost = Gamma[0]
        Gamma = Gamma[1:]

        nx, ny = ni
        Gamma_n = np.linalg.pinv(M) @ (k1_i*np.array([0,nx,ny,0,0,0]))
        Gamma_n_ghost = Gamma_n[0]
        Gamma_n = Gamma_n[1:]
        Gg = Gamma_ghost / Gamma_n_ghost
        beta_i = beta(p[i])
        
        K[i,I1] += Gamma - Gg * Gamma_n
        F[i] += F[i].toarray() + source(p[i]) - Gg * beta_i


def Ln_gen(p,b,i):
    n = normal_vectors(b,p)
    ni = n[b==i][0]
    nx, ny = ni
    Ln = np.array([0,nx,ny,0,0,0])
    return Ln

def neumann_assembling(p,b,triangles,k,source,L,u_n,K,F,Ln_gen=Ln_gen):
    n = normal_vectors(b,p)
    for i in b:
        I = support_nodes(i,triangles)
        ni = n[b==i][0]

        deltasx = p[I,0] - p[i,0]
        deltasy = p[I,1] - p[i,1]
        ghost = np.array([-np.mean(deltasx), -np.mean(deltasy)])
        norm_ghost = np.linalg.norm(ghost)
        ghostx, ghosty = norm_ghost * ni

        norm_ghost = np.linalg.norm(np.array([ghostx, ghosty]))
        mean_delta = np.mean(np.sqrt(deltasx**2 + deltasy**2)[1:])
        scale_factor = mean_delta/norm_ghost
        ghostx = scale_factor * ghostx
        ghosty = scale_factor * ghosty

        deltasx = np.hstack((ghostx, deltasx))
        deltasy = np.hstack((ghosty, deltasy))

        M = deltas_matrix(deltasx, deltasy)
        k_i = k(p[i])
        Gamma = np.linalg.pinv(M) @ (k_i*L)
        Gamma_ghost = Gamma[0]
        Gamma = Gamma[1:]

        nx, ny = ni
        Ln = Ln_gen(p,b,i)
        Gamma_n = np.linalg.pinv(M) @ (k_i*Ln)
        Gamma_n_ghost = Gamma_n[0]
        Gamma_n = Gamma_n[1:]
        Gg = Gamma_ghost / Gamma_n_ghost
        K[i,I] = Gamma - Gg * Gamma_n
        F[i] = F[i].toarray() + source(p[i]) - Gg * u_n(p[i])
    return K, F



def create_system_K_F(
        p,
        triangles,
        L,
        source,
        materials,
        neumann_boundaries,
        dirichlet_boundaries,
        interfaces={},
        Ln_gen=Ln_gen
    ):
    """
    Assembles `K` and `F` for system  `KU=F`.

    Parameters
    ----------
    p : numpy.ndarray
        2D coordinates of all domain nodes, with shape (n_nodes,2).
    triangles : numpy.ndarray
        Array with shape (n,3), containing index of the n triangles
        with 3 nodes each.
    L : numpy.ndarray
        Coefitiens vector `L` = [A,B,C,D,E,F] of GFDM system `M\Gamma=L`,
        where each entrance multiplies each term of the linear differential
        operator
        `\mathb{L} = Au + Bu_x + Cu_y + Du_{xx} + Eu_{xy} + Fu_{yy}`.
    source : function
        Implemented for one node coordinates, e.g. `source = lambda p: p[0] + p[1]`.
    materials : dict
        Material properties, e.g. `materials["0"] = [k, interior]`, where `k` is
        the permeability function, `interior` are index for interior nodes.
    neumann_boundaries : dict
        Neumann properties, e.g. `neumann_boundaries["bottom"] = [k, nodesb, fNeu]`,
        where `k` is the permeability function, `nodesb` are index for
        neumann boundary nodes, `fNeu` is a function for neumann prescribed values.
    dirichlet_boundaries : dict
        Dirichlet properties, e.g. `dirichlet_boundaries["left"] = [nodesl, fDir]`,
        where `nodesl` are index for dirichlet boundary nodes, `fDir` is a
        function for dirichlet prescribed values.
    interfaces : dict, optional
        Interfaces properties, e.g. `interfaces["A"] = [k0, k1, b, beta, m0, m1]`,
        where `k0` is permeability of material0, `k1`is permeability of material1,
        `beta` is source at the interface, `m0` are index of material0 nodes,
        `m1` are index of material1 nodes. The default is {}.

    Returns
    -------
    K : scipy.sparse._csr.csr_matrix
        Stiffness matrix of system `KU=F`.
    F : numpy.ndarray
        Vector of forces of system `KU=F`.

    """
    L[3] *= 2
    L[5] *= 2

    N = p.shape[0]
    K = sp.lil_matrix((N,N))
    F = sp.lil_matrix((N,1))

    # Interior nodes
    # stacking neumann boundaries as an array
    bN = np.array([])
    for key in neumann_boundaries:
        b = neumann_boundaries[key][1]
        bN = np.hstack((bN, b))
    
    # cicle through interior nodes
    for material in materials:
        k = materials[material][0]
        b = materials[material][1]
        b = np.setdiff1d(b,bN)             # interior nodes
        K, F = interior_assembling(p,b,triangles,L,k,source,K,F)

    # Neumman boundaries
    for boundary in neumann_boundaries:
        k = neumann_boundaries[boundary][0]
        b = neumann_boundaries[boundary][1]
        u_n = neumann_boundaries[boundary][2]
        K, F = neumann_assembling(p,b,triangles,k,source,L,u_n,K,F,Ln_gen)

    # Interfaces
    for interface in interfaces:
        k0 = interfaces[interface][0]
        k1 = interfaces[interface][1]
        b = interfaces[interface][2]
        beta = interfaces[interface][3]
        m0 = interfaces[interface][4]
        m1 = interfaces[interface][5]

        K = sp.lil_matrix(K)
        F = sp.lil_matrix(F)
        
        # interface in material 0
        K, F = interface_assembling(p,b,triangles,m1,m0,k0,k1,source,L,beta,K,F)
            
    # Dirichlet boundaries
    K = sp.lil_matrix(K)
    F = sp.lil_matrix(F)
    for boundary in dirichlet_boundaries:
        b = dirichlet_boundaries[boundary][0]
        u = dirichlet_boundaries[boundary][1]
        K, F = dirichlet_assembling(p,b,u,K,F)

    K = sp.csr_matrix(K)
    F = F.toarray().flatten()

    return K, F