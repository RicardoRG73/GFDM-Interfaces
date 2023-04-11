import numpy as np

def support_nodes(i,triangles):
    temp =  np.any( np.isin(triangles,i), axis=1)
    temp = triangles[temp,:].flatten()
    I = np.setdiff1d(temp,i)
    I = np.hstack((i,I))
    return I

def normal_vectors(b,p):
    percentage_line_tolerance = 0.99
    N = b.shape[0]
    l1 = p[b[1],:] - p[b[0],:]
    l1 = l1 / np.linalg.norm(l1)
    l2 = p[b[b.shape[0]//2],:] - p[b[0],:]
    l2 = l2 / np.linalg.norm(l2)
    line = np.dot(l1,l2) > percentage_line_tolerance
    rotation = np.array([[0,-1],[1,0]])
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

    N = p.shape[0]
    K = np.zeros((N,N))
    F = np.zeros((N,))

    # Interior nodes
    for material in materials:
        k = materials[material][0]
        b = materials[material][1]
        for i in b:
            I = support_nodes(i,triangles)
            deltas_x = p[i,0] - p[I,0]
            deltas_y = p[i,1] - p[I,1]
            M = np.vstack((
                np.ones(deltas_x.shape),
                deltas_x,
                deltas_y,
                deltas_x**2,
                deltas_x*deltas_y,
                deltas_y**2
            ))
            Gamma = np.linalg.pinv(M) @ (k*L)
            K[i,I] += Gamma
            F[i] += source(p[i])

    # Neumman boundaries
    for boundary in neumann_boundaries:
        k = neumann_boundaries[boundary][0]
        b = neumann_boundaries[boundary][1]
        u_n = neumann_boundaries[boundary][2]
        n = normal_vectors(b,p)
        for i in b:
            I = support_nodes(i,triangles)
            ni = n[b==i][0]

            deltas_x = p[i,0] - p[I,0]
            deltas_y = p[i,1] - p[I,1]
            ghost = np.array([-np.mean(deltas_x), -np.mean(deltas_y)])
            dot_ghost_n = ghost @ ni
            ghost_x, ghost_y = dot_ghost_n * ghost

            deltas_x = np.hstack((ghost_x, deltas_x))
            deltas_y = np.hstack((ghost_y, deltas_y))

            M = np.vstack((
                np.ones(deltas_x.shape),
                deltas_x,
                deltas_y,
                deltas_x**2,
                deltas_x*deltas_y,
                deltas_y**2
            ))
            Gamma = np.linalg.pinv(M) @ (k*L)
            Gamma_ghost = Gamma[0]
            Gamma = Gamma[1:]

            nx, ny = ni
            Gamma_n = np.linalg.pinv(M) @ (k*[0,nx,ny,0,0,0])
            Gamma_n_ghost = Gamma_n[0]
            Gamma_n = Gamma_n[1:]
            Gg = Gamma_ghost / Gamma_n_ghost
            K[i,I] += Gamma - Gg * Gamma_n
            F[i] += source(p[i]) - Gg * u_n(p[i])

    # Interfaces
    for interface in interfaces:
        k0 = interfaces[interface][0]
        b0 = interfaces[interface][1]
        f0 = interfaces[interface][2]
        k1 = interfaces[interface][3]
        b1 = interfaces[interface][4]
        f1 = interfaces[interface][5] 
        n = normal_vectors(b0,p)
        if b0.shape[0] == b1.shape[0]:
            for i in b0:
                I = support_nodes(i,triangles)
                ni = n[i==b0][0]
                deltas_x = p[i,0] - p[I,0]
                deltas_y = p[i,1] - p[I,1]
                ghost = np.array([-np.mean(deltas_x), -np.mean(deltas_y)])
                dot_ghost_n = ghost @ ni
                ghost_x, ghost_y = dot_ghost_n * ghost

                deltas_x = np.hstack((ghost_x, deltas_x))
                deltas_y = np.hstack((ghost_y, deltas_y))

                M = np.vstack((
                    np.ones(deltas_x.shape),
                    deltas_x,
                    deltas_y,
                    deltas_x**2,
                    deltas_x*deltas_y,
                    deltas_y**2
                ))
                Gamma = np.linalg.pinv(M) @ (k0*L)
                Gamma_ghost = Gamma[0]
                Gamma = Gamma[1:]

                nx, ny = ni
                Gamma_n = np.linalg.pinv(M) @ (k0*[0,nx,ny,0,0,0])
                Gamma_n_ghost = Gamma_n[0]
                Gamma_n = Gamma_n[1:]
                Gg = Gamma_ghost / Gamma_n_ghost
                K[i,I] += Gamma - Gg * Gamma_n
                F[i] += source(p[i]) - Gg * f0(p[i]) / 2

            for i in b1:
                I = support_nodes(i,triangles)
                ni = n[i==b1][0]
                deltas_x = p[i,0] - p[I,0]
                deltas_y = p[i,1] - p[I,1]
                ghost = np.array([-np.mean(deltas_x), -np.mean(deltas_y)])
                dot_ghost_n = ghost @ ni
                ghost_x, ghost_y = dot_ghost_n * ghost

                deltas_x = np.hstack((ghost_x, deltas_x))
                deltas_y = np.hstack((ghost_y, deltas_y))

                M = np.vstack((
                    np.ones(deltas_x.shape),
                    deltas_x,
                    deltas_y,
                    deltas_x**2,
                    deltas_x*deltas_y,
                    deltas_y**2
                ))
                Gamma = np.linalg.pinv(M) @ (k1*L)
                Gamma_ghost = Gamma[0]
                Gamma = Gamma[1:]

                nx, ny = ni
                Gamma_n = np.linalg.pinv(M) @ (k1*[0,nx,ny,0,0,0])
                Gamma_n_ghost = Gamma_n[0]
                Gamma_n = Gamma_n[1:]
                Gg = Gamma_ghost / Gamma_n_ghost
                K[i,I] += Gamma - Gg * Gamma_n
                F[i] += source(p[i]) - Gg * f1(p[i]) / 2

        # solution discontinuity implementation
        # N_interface = b0.shape[0]
        # K = np.vstack((K,np.zeros((N_interface,N))))
        # F = np.hstack((F,np.zeros(N_interface)))
        # K[N:,b0] = np.eye(N_interface)
        # K[N:,b1] = -np.eye(N_interface)
                
                

    # Dirichlet boundaries
    for boundary in dirichlet_boundaries:
        b = dirichlet_boundaries[boundary][0]
        u = dirichlet_boundaries[boundary][1]
        for i in b:
            F -= K[:,i] * u(p[i])
            F[i] = u(p[i])
            K[:,i] = 0
            K[i,i] = 1



    return K,F