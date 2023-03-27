import numpy as np

def support_nodes(i,triangles):
    temp =  np.any( np.isin(triangles,i), axis=1)
    t = triangles[temp,:]
    t = t.flatten()
    I = np.setdiff1d(t,i)
    I = np.hstack((i,I))
    return I


def create_system_K_F(
        p,
        material_properties,
        triangles
    ):
    N = p.shape[0]
    K = np.zeros((N,N))
    F = np.zeros((N,))

    # Interior nodes
    for key in material_properties:
        k = material_properties[key][0]
        bm = material_properties[key][1]
        for i in bm:
            I = support_nodes(i,triangles.copy())
            horizontal = p[i,0] - p[I,0]
            vertical = p[i,1] - p[I,1]
            
            K[i,i] = 1
            F[i] = 1

    return K,F