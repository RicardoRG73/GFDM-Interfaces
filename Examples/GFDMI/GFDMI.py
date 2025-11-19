import numpy as np
import scipy.sparse as sp

class GFDMI_2D_problem:
    def __init__(self, coords, triangles, L, source):
        self.coords = coords
        self.triangles = triangles
        self.L = L
        self.source = source
        self.materials = {}
        self.neumann_boundaries = {}
        self.dirichlet_boundaries = {}
        self.interfaces = {}
    
    @staticmethod
    def support_nodes(node, triangles, min_support_nodes=5, max_iter=2):
        """
        Returns the index of support nodes `I` corresponding to the central node
        with index `node`. 
        
        Parameters
        ----------
        node : int
            index of central node.
        triangles : numpy.ndarray
            array with shape (n,3), containing index of the n triangles with 3 nodes each.
        min_support_nodes : int, optional
            number of minimum support nodes. The default is 5.
        max_iter : int, optional
            number of maximum iterations for adding support nodes to the list `I`. The default is 2.

        Returns
        -------
        support_nodes : numpy.ndarray
            index of the support nodes of central `node`.
        """
        support_nodes = {node}  # Use a set for unique support nodes
        iter_count = 0

        while len(support_nodes) < min_support_nodes and iter_count < max_iter:
            # Find triangles containing the current support nodes
            temp = np.any(np.isin(
                triangles,
                list(support_nodes)
            ), axis=1)
            support_nodes.update(triangles[temp].flatten())  # Add new nodes to the set
            iter_count += 1

        return np.array(list(support_nodes))  # Convert back to array before returning

    def normal_vectors(self,boundary_nodes):
        """
        Computes normal vectors at boundary nodes.

        Parameters
        ----------
        boundary_nodes : numpy.ndarray
            index of boundary nodes.
        coords : numpy.ndarray
            array with shape (n,2) containing the coordinates of the n nodes.

        Returns
        -------
        normal_vecs : numpy.ndarray
            array with shape (N,2) containing the normal vectors at the N boundary nodes.
        """
        coords = self.coords
        line_tolerance = 0.99
        
        N = boundary_nodes.shape[0]         # number of boundary nodes
        
        line_1 = coords[boundary_nodes[1],:] - coords[boundary_nodes[0],:]
        line_1 = line_1 / np.linalg.norm(line_1)
        line_2 = coords[boundary_nodes[N//2],:] - coords[boundary_nodes[0],:]
        line_2 = line_2 / np.linalg.norm(line_2)
        is_line = np.dot(line_1,line_2) > line_tolerance
        
        clockwise_rotation = np.array([[0,1],[-1,0]])
        
        if is_line:
            line_1 = clockwise_rotation @ line_1
            normal_vecs = np.kron(np.ones(N),line_1).reshape((N,2))

        else:
            normal_vecs = np.zeros((N,2))
            centroid = np.mean(coords,axis=0)

            for node in boundary_nodes:
                distance = np.sqrt((coords[node,0]-coords[boundary_nodes,0])**2 + (coords[node,1]-coords[boundary_nodes,1])**2)
                closest_nodes = boundary_nodes[distance.argsort()[:7]]
                closest_centroid = np.mean(coords[closest_nodes,:], axis=0)
                v1 = coords[closest_nodes[5]] - closest_centroid
                v2 = coords[closest_nodes[6]] - closest_centroid
                ni = clockwise_rotation @ (v2-v1) / np.linalg.norm(v2-v1)
                ni = ni * np.dot(ni , coords[node]-centroid)
                ni = ni / np.linalg.norm(ni)
                normal_vecs[boundary_nodes==node] = ni
        
        return normal_vecs
    
    def add_material(self,label,permeability,interior_nodes):
        self.materials[label] = [permeability, interior_nodes]

    def add_neumann_boundary(self,label,permeability,boundary_nodes,condition):
        self.neumann_boundaries[label] = [permeability, boundary_nodes, condition]

    def add_dirichlet_boundary(self,label,boundary_nodes,condition):
        self.dirichlet_boundaries[label] = [boundary_nodes, condition]

    def add_interface(
            self,
            label,
            permeability_left_mat,
            permeability_right_mat,
            left_interface_nodes,
            right_interface_nodes,
            flux_difference_beta,
            solution_difference_alpha,
            interior_left_nodes,
            interior_right_nodes
        ):
        self.interfaces[label] = [
            permeability_left_mat,
            permeability_right_mat,
            left_interface_nodes,
            right_interface_nodes,
            flux_difference_beta,
            solution_difference_alpha,
            interior_left_nodes,
            interior_right_nodes
        ]

    def create_system_K_F(self):
        """ 
        Assembles `K` and `F` for system  `KU=F`
        """
        coords = self.coords
        triangles = self.triangles
        L = self.L
        source = self.source
        materials = self.materials
        neumann_boundaries = self.neumann_boundaries
        dirichlet_boundaries = self.dirichlet_boundaries
        interfaces = self.interfaces

        L[3] *= 2
        L[5] *= 2

        N = coords.shape[0]
        K = sp.lil_matrix((N,N))
        F = sp.lil_matrix((N,1))

        # Interior nodes
        bns = np.array([])                      # all nodes in neumann boundaries
        for key in neumann_boundaries:
            b = neumann_boundaries[key][1]
            bns = np.hstack((bns, b))

        for material in materials:
            k = materials[material][0]
            b = materials[material][1]
            b = np.setdiff1d(b,bns)             # interior nodes
            for i in b:
                I = GFDMI_2D_problem.support_nodes(i,triangles)
                deltasx = coords[I,0] - coords[i,0]
                deltasy = coords[I,1] - coords[i,1]
                M = np.vstack((
                    np.ones(deltasx.shape),
                    deltasx,
                    deltasy,
                    deltasx**2,
                    deltasx*deltasy,
                    deltasy**2
                ))
                Gamma = np.linalg.pinv(M) @ (k(coords[i])*L)
                K[i,I] = Gamma
                F[i] = source(coords[i])

        # Neumman boundaries
        for boundary in neumann_boundaries:
            k = neumann_boundaries[boundary][0]
            b = neumann_boundaries[boundary][1]
            u_n = neumann_boundaries[boundary][2]
            n = self.normal_vectors(b)
            for i in b:
                I = self.support_nodes(i,triangles)
                ni = n[b==i][0]

                deltasx = coords[I,0] - coords[i,0]
                deltasy = coords[I,1] - coords[i,1]
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

                M = np.vstack((
                    np.ones(deltasx.shape),
                    deltasx,
                    deltasy,
                    deltasx**2,
                    deltasx*deltasy,
                    deltasy**2
                ))
                k_i = k(coords[i])
                Gamma = np.linalg.pinv(M) @ (k_i*L)
                Gamma_ghost = Gamma[0]
                Gamma = Gamma[1:]

                nx, ny = ni
                Gamma_n = np.linalg.pinv(M) @ (k_i*np.array([0,nx,ny,0,0,0]))
                Gamma_n_ghost = Gamma_n[0]
                Gamma_n = Gamma_n[1:]
                Gg = Gamma_ghost / Gamma_n_ghost
                K[i,I] = Gamma - Gg * Gamma_n
                F[i] = F[i].toarray() + source(coords[i]) - Gg * u_n(coords[i])

        # Interfaces
        for interface in interfaces:
            k0 = interfaces[interface][0]
            k1 = interfaces[interface][1]
            biA = interfaces[interface][2]
            biB = interfaces[interface][3]
            beta = interfaces[interface][4]
            alpha = interfaces[interface][5]
            m0 = interfaces[interface][6]
            m1 = interfaces[interface][7]

            K = sp.lil_matrix(K)
            F = sp.lil_matrix(F)
            
            # interface in material 0
            n = self.normal_vectors(biA)
            for i in biA:
                I0 = self.support_nodes(i,triangles)
                I0 = np.setdiff1d(I0,m1)

                ni = n[biA==i][0]

                deltasx = coords[I0,0] - coords[i,0]
                deltasy = coords[I0,1] - coords[i,1]
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

                M = np.vstack((
                    np.ones(deltasx.shape),
                    deltasx,
                    deltasy,
                    deltasx**2,
                    deltasx*deltasy,
                    deltasy**2
                ))
                k0_i = k0(coords[i])
                Gamma = np.linalg.pinv(M) @ (k0_i*L)
                Gamma_ghost = Gamma[0]
                Gamma = Gamma[1:]

                nx, ny = ni
                Gamma_n = np.linalg.pinv(M) @ (k0_i*np.array([0,nx,ny,0,0,0]))
                Gamma_n_ghost = Gamma_n[0]
                Gamma_n = Gamma_n[1:]
                Gg = Gamma_ghost / Gamma_n_ghost
                beta_i = beta(coords[i])
                K[i,I0] = Gamma - Gg * Gamma_n
                F[i] = source(coords[i]) - Gg * beta_i
            
            # interface in material 1
            n = self.normal_vectors(biB)
            for i in biB:
                I0 = self.support_nodes(i,triangles)
                I0 = np.setdiff1d(I0,m0)

                ni = -n[biB==i][0]

                deltasx = coords[I0,0] - coords[i,0]
                deltasy = coords[I0,1] - coords[i,1]
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

                M = np.vstack((
                    np.ones(deltasx.shape),
                    deltasx,
                    deltasy,
                    deltasx**2,
                    deltasx*deltasy,
                    deltasy**2
                ))
                k1_i = k1(coords[i])
                Gamma = np.linalg.pinv(M) @ (k1_i*L)
                Gamma_ghost = Gamma[0]
                Gamma = Gamma[1:]

                nx, ny = ni
                Gamma_n = np.linalg.pinv(M) @ (k1_i*np.array([0,nx,ny,0,0,0]))
                Gamma_n_ghost = Gamma_n[0]
                Gamma_n = Gamma_n[1:]
                Gg = Gamma_ghost / Gamma_n_ghost
                beta_i = beta(coords[i])
                
                biA_i = biA[np.argmin(
                    np.sqrt(
                        (coords[biA,0]-coords[i,0])**2 + (coords[biA,1]-coords[i,1])**2
                    )
                )]
                K[biA_i,I0] = K[biA_i,I0].toarray() + Gamma - Gg * Gamma_n
                F[biA_i] = F[biA_i].toarray() + source(coords[i]) - Gg * beta_i
                K[i,biA_i] = -1
                K[i,i] = 1
                F[i] = alpha(coords[i,:])
                
        # Dirichlet boundaries
        K = sp.lil_matrix(K)
        F = sp.lil_matrix(F)
        for boundary in dirichlet_boundaries:
            b = dirichlet_boundaries[boundary][0]
            u = dirichlet_boundaries[boundary][1]
            for i in b:
                # F -= K[:,i] * u(p[i])
                F[i] = u(coords[i])
                # K[:,i] = 0
                K[i,i] = 1

        K = sp.csr_matrix(K)
        F = F.toarray().flatten()

        return K, F

    def create_system_K_F_cont_U(self):
        """ 
        Assembles `K` and `F` for system  `KU=F`, where `U` is continuos
        """
        coords = self.coords
        triangles = self.triangles
        L = self.L
        source = self.source

        materials = self.materials
        neumann_boundaries = self.neumann_boundaries
        dirichlet_boundaries = self.dirichlet_boundaries
        interfaces = self.interfaces

        L[3] *= 2
        L[5] *= 2

        N = coords.shape[0]
        K = sp.lil_matrix((N,N))
        F = sp.lil_matrix((N,1))

        # Interior nodes
        bns = np.array([])                      # all nodes in neumann boundaries
        for key in neumann_boundaries:
            b = neumann_boundaries[key][1]
            bns = np.hstack((bns, b))

        for material in materials:
            k = materials[material][0]
            b = materials[material][1]
            b = np.setdiff1d(b,bns)             # interior nodes
            for i in b:
                I = GFDMI_2D_problem.support_nodes(i,triangles)
                deltasx = coords[I,0] - coords[i,0]
                deltasy = coords[I,1] - coords[i,1]
                M = np.vstack((
                    np.ones(deltasx.shape),
                    deltasx,
                    deltasy,
                    deltasx**2,
                    deltasx*deltasy,
                    deltasy**2
                ))
                Gamma = np.linalg.pinv(M) @ (k(coords[i])*L)
                K[i,I] = Gamma
                F[i] = source(coords[i])

        # Neumman boundaries
        for boundary in neumann_boundaries:
            k = neumann_boundaries[boundary][0]
            b = neumann_boundaries[boundary][1]
            u_n = neumann_boundaries[boundary][2]
            n = self.normal_vectors(b)
            for i in b:
                I = GFDMI_2D_problem.support_nodes(i,triangles)
                ni = n[b==i][0]

                deltasx = coords[I,0] - coords[i,0]
                deltasy = coords[I,1] - coords[i,1]
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

                M = np.vstack((
                    np.ones(deltasx.shape),
                    deltasx,
                    deltasy,
                    deltasx**2,
                    deltasx*deltasy,
                    deltasy**2
                ))
                k_i = k(coords[i])
                Gamma = np.linalg.pinv(M) @ (k_i*L)
                Gamma_ghost = Gamma[0]
                Gamma = Gamma[1:]

                nx, ny = ni
                Gamma_n = np.linalg.pinv(M) @ (k_i*np.array([0,nx,ny,0,0,0]))
                Gamma_n_ghost = Gamma_n[0]
                Gamma_n = Gamma_n[1:]
                Gg = Gamma_ghost / Gamma_n_ghost
                K[i,I] = Gamma - Gg * Gamma_n
                F[i] = F[i].toarray() + source(coords[i]) - Gg * u_n(coords[i])

        # Interfaces
        for interface in interfaces:
            k0 = interfaces[interface][0]
            k1 = interfaces[interface][1]
            b = interfaces[interface][2]
            _ = interfaces[interface][3]
            beta = interfaces[interface][4]
            _ = interfaces[interface][5]
            m0 = interfaces[interface][6]
            m1 = interfaces[interface][7]

            K = sp.lil_matrix(K)
            F = sp.lil_matrix(F)
            
            # material 0
            n = self.normal_vectors(b)
            for i in b:
                I0 = GFDMI_2D_problem.support_nodes(i,triangles)
                I0 = np.setdiff1d(I0,m1)

                ni = n[b==i][0]

                deltasx = coords[I0,0] - coords[i,0]
                deltasy = coords[I0,1] - coords[i,1]
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

                M = np.vstack((
                    np.ones(deltasx.shape),
                    deltasx,
                    deltasy,
                    deltasx**2,
                    deltasx*deltasy,
                    deltasy**2
                ))
                k0_i = k0(coords[i])
                Gamma = np.linalg.pinv(M) @ (k0_i*L)
                Gamma_ghost = Gamma[0]
                Gamma = Gamma[1:]

                nx, ny = ni
                Gamma_n = np.linalg.pinv(M) @ (k0_i*np.array([0,nx,ny,0,0,0]))
                Gamma_n_ghost = Gamma_n[0]
                Gamma_n = Gamma_n[1:]
                Gg = Gamma_ghost / Gamma_n_ghost
                beta_i = beta(coords[i])
                K[i,I0] = Gamma - Gg * Gamma_n
                F[i] = source(coords[i]) - Gg * beta_i
            
            # material 1
            for i in b:
                I1 = GFDMI_2D_problem.support_nodes(i,triangles)
                I1 = np.setdiff1d(I1,m0)

                ni = -n[b==i][0]

                deltasx = coords[I1,0] - coords[i,0]
                deltasy = coords[I1,1] - coords[i,1]
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

                M = np.vstack((
                    np.ones(deltasx.shape),
                    deltasx,
                    deltasy,
                    deltasx**2,
                    deltasx*deltasy,
                    deltasy**2
                ))
                k1_i = k1(coords[i])
                Gamma = np.linalg.pinv(M) @ (k1_i*L)
                Gamma_ghost = Gamma[0]
                Gamma = Gamma[1:]

                nx, ny = ni
                Gamma_n = np.linalg.pinv(M) @ (k1_i*np.array([0,nx,ny,0,0,0]))
                Gamma_n_ghost = Gamma_n[0]
                Gamma_n = Gamma_n[1:]
                Gg = Gamma_ghost / Gamma_n_ghost
                beta_i = beta(coords[i])
                
                K[i,I1] = K[i,I1].toarray() + Gamma - Gg * Gamma_n
                F[i] = F[i].toarray() + source(coords[i]) - Gg * beta_i
                
        # Dirichlet boundaries
        K = sp.lil_matrix(K)
        F = sp.lil_matrix(F)
        for boundary in dirichlet_boundaries:
            b = dirichlet_boundaries[boundary][0]
            u = dirichlet_boundaries[boundary][1]
            for i in b:
                # F -= K[:,i] * u(p[i])
                F[i] = u(coords[i])
                # K[:,i] = 0
                K[i,i] = 1

        K = sp.csr_matrix(K)
        F = F.toarray().flatten()

        return K, F