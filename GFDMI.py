import numpy as np
import scipy.sparse as sp

def support_nodes(i,triangles, min_support_nodes=5, max_iter=2):
    I = np.array([i])
    iter = 1
    while I.shape[0] < min_support_nodes and iter <= max_iter:
        temp =  np.any( np.isin(triangles,I), axis=1)
        temp = triangles[temp,:].flatten()
        I = np.setdiff1d(temp,i)
        I = np.hstack((i,I))
        iter += 1
    return I

def normal_vectors(b,p):
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
            imin = b[d.argsort()[:3]]
            pm = np.mean(p[imin,:], axis=0)
            v1 = p[imin[1]] - pm
            v2 = p[imin[2]] - pm
            ni = rotation @ (v2-v1) / np.linalg.norm(v2-v1)
            ni = ni * np.dot(ni , p[i]-centroid)
            ni = ni / np.linalg.norm(ni)
            n[b==i] = ni
    
    return n


def create_system_K_F(
        p,
        triangles,
        L,
        source,
        materials,
        neumann_boundaries,
        dirichlet_boundaries,
        interfaces={}
    ):
    """ 
    Assembles `K` and `F` for system  `KU=F`
    """
    L[3] *= 2
    L[5] *= 2

    N = p.shape[0]
    K = sp.lil_matrix((N,N))
    F = sp.lil_matrix((N,1))

    # Interior nodes
    bns = np.array([])
    for key in neumann_boundaries:
        b = neumann_boundaries[key][1]
        bns = np.hstack((bns, b))

    for material in materials:
        k = materials[material][0]
        b = materials[material][1]
        b = np.setdiff1d(b,bns)
        for i in b:
            I = support_nodes(i,triangles)
            deltasx = p[I,0] - p[i,0]
            deltasy = p[I,1] - p[i,1]
            M = np.vstack((
                np.ones(deltasx.shape),
                deltasx,
                deltasy,
                deltasx**2,
                deltasx*deltasy,
                deltasy**2
            ))
            Gamma = np.linalg.pinv(M) @ (k(p[i])*L)
            K[i,I] = Gamma
            F[i] = source(p[i])

    # Neumman boundaries
    for boundary in neumann_boundaries:
        k = neumann_boundaries[boundary][0]
        b = neumann_boundaries[boundary][1]
        u_n = neumann_boundaries[boundary][2]
        n = normal_vectors(b,p)
        for i in b:
            I = support_nodes(i,triangles)
            ni = n[b==i][0]

            deltasx = p[I,0] - p[i,0]
            deltasy = p[I,1] - p[i,1]
            ghost = np.array([-np.mean(deltasx), -np.mean(deltasy)])
            norm_ghost = I.shape[0] * np.linalg.norm(ghost)
            ghostx, ghosty = norm_ghost * ghost

            deltasx = np.hstack((ghostx, deltasx))
            deltasy = np.hstack((ghosty, deltasy))

            M = np.vstack((
                np.ones(deltasx.shape),
                deltasx,
                deltasy,
                deltasx**2,
                deltasx*deltasy,
                deltasy**2
            ))
            k_i = k(p[i])
            Gamma = np.linalg.pinv(M) @ (L)
            Gamma_ghost = Gamma[0]
            Gamma = Gamma[1:]

            nx, ny = ni
            Gamma_n = np.linalg.pinv(M) @ (np.array([0,nx,ny,0,0,0]))
            Gamma_n_ghost = Gamma_n[0]
            Gamma_n = Gamma_n[1:]
            Gg = Gamma_ghost / Gamma_n_ghost
            K[i,I] = k_i * (Gamma - Gg * Gamma_n)
            F[i] = F[i].toarray() + source(p[i]) - k_i * Gg * u_n(p[i])

    # Interfaces
    # K = sp.csr_matrix(K)
    for interface in interfaces:
        k0 = interfaces[interface][0]
        k1 = interfaces[interface][1]
        b = interfaces[interface][2]
        beta = interfaces[interface][3]
        alpha = interfaces[interface][4]
        material0 = interfaces[interface][5]
        material1 = interfaces[interface][6]
        m0 = materials[material0][1]
        m1 = materials[material1][1]
        n = normal_vectors(b,p)

        m = b.shape[0]                              # m nodes at interfaace
        N = p.shape[0]                              # N total nodes in the domain

        # Double nodes
        p = np.vstack((p, p[b,:]))
        #   K modification
        K = sp.hstack((K, sp.lil_matrix((N,m))))
        K = sp.vstack((K, sp.lil_matrix((m,N+m))))
        K = sp.lil_matrix(K)

        bd = np.arange(N,N+m)                       # index for double nodes

        K[bd,bd] = 1
        K[bd,b] = -1

        #   F modification
        F = sp.vstack((F, sp.lil_matrix((m,1))))
        F = sp.lil_matrix(F)
        for i in bd:
            F[i] = alpha(p[i])

        i2 = N
        for i in b:
            # Material M0, whit b interface-original-nodes
            I_all = support_nodes(i,triangles)
            I = np.setdiff1d(I_all, m1)
            ni = n[b==i][0]

            deltasx = p[I,0] - p[i,0]
            deltasy = p[I,1] - p[i,1]
            ghost = np.array([-np.mean(deltasx), -np.mean(deltasy)])
            norm_ghost = I.shape[0] * np.linalg.norm(ghost)
            ghostx, ghosty = norm_ghost * ni

            deltasx = np.hstack((ghostx, deltasx))
            deltasy = np.hstack((ghosty, deltasy))

            M = np.vstack((
                np.ones(deltasx.shape),
                deltasx,
                deltasy,
                deltasx**2,
                deltasx*deltasy,
                deltasy**2
            ))
            k0_i = k0(p[i])
            Gamma = np.linalg.pinv(M) @ (L)
            Gamma_ghost = Gamma[0]
            Gamma = Gamma[1:]

            nx, ny = ni
            Gamma_n = np.linalg.pinv(M) @ (np.array([0,nx,ny,0,0,0]))
            Gamma_n_ghost = Gamma_n[0]
            Gamma_n = Gamma_n[1:]
            Gg = Gamma_ghost / Gamma_n_ghost
            beta_i = beta(p[i])
            K[i,I] = k0_i * (Gamma - Gg * Gamma_n)
            F[i] = source(p[i]) - k0_i * Gg * beta_i

            # Material M1, with b_d interface-double-nodes
            I = np.setdiff1d(I_all, m0)
            deltasx = p[I,0] - p[i,0]
            deltasy = p[I,1] - p[i,1]
            ghost = np.array([-np.mean(deltasx), -np.mean(deltasy)])
            norm_ghost = I.shape[0] * np.linalg.norm(ghost)
            ghostx, ghosty = - norm_ghost * ni

            deltasx = np.hstack((ghostx, deltasx))
            deltasy = np.hstack((ghosty, deltasy))

            M = np.vstack((
                np.ones(deltasx.shape),
                deltasx,
                deltasy,
                deltasx**2,
                deltasx*deltasy,
                deltasy**2
            ))
            k1_i = k1(p[i])
            Gamma = np.linalg.pinv(M) @ (L)
            Gamma_ghost = Gamma[0]
            Gamma = Gamma[1:]

            nx, ny = ni
            Gamma_n = np.linalg.pinv(M) @ (np.array([0,nx,ny,0,0,0]))
            Gamma_n_ghost = Gamma_n[0]
            Gamma_n = Gamma_n[1:]
            Gg = Gamma_ghost / Gamma_n_ghost

            # double nodes modification
            Ib = np.setdiff1d(I,m1)                         # index for nodes at interface
            Ib = np.setdiff1d(Ib,I[0])                      # leaving out the central node
            for j in Ib:
                dist = np.sum((p[j] - p[bd])**2, axis=1)
                I[I==Ib] = bd[dist==0]
                
            I[0] = i2

            md = support_nodes(i, triangles)
            
            
            beta_i = beta(p[i])
            K[i,I] += k1_i * (Gamma - Gg * Gamma_n)
            F[i] = F[i].toarray() + source(p[i]) - k1_i * Gg * beta_i
            i2 += 1
                

    # Dirichlet boundaries
    K = sp.lil_matrix(K)
    F = sp.lil_matrix(F)
    for boundary in dirichlet_boundaries:
        b = dirichlet_boundaries[boundary][0]
        u = dirichlet_boundaries[boundary][1]
        for i in b:
            F -= K[:,i] * u(p[i])
            F[i] = u(p[i])
            K[:,i] = 0
            K[i,i] = 1

    K = sp.csr_matrix(K)
    F = sp.csr_matrix(F)

    U = sp.linalg.spsolve(K,F)

    return K, F, U, p