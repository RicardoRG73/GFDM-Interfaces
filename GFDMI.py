import numpy as np

def interface_normal_vec(p, i, b):
    """Calculates normal vector components `nx` and `ny`
    for a central node `i` given, located in the interface `b`.
    Parameters
    -----
    p : array_like
        Node coordinates, ordered by rows p[0] = np.array([x0, y0]).
    i : int
        Central node index.
    b : ndarray
        Interface nodes.
    Returns
    -----
    n : ndarray
        Normal vector.
    """
    tol = 0.001                                                 # tolerance to check if boundary is a line
    p0 = p[b[0]]                                                # coords of first boundary node
    p1 = p[b[1]]                                                # coords of last boundary node
    pm = p[b[len(b)//2]]                                        # coords of boundary mid-node
    pmm1 = p[b[len(b)//4]]
    pmm2 = p[b[3*len(b)//4]]
    n01 = (p1 - p0) / np.linalg.norm(p1 - p0)                            # normal vector in direction p0-p1
    n0m = (pm - p0) / np.linalg.norm(pm - p0)                            # normal vector in direction p0-pm
    n0m1 = (pmm1 - p0) / np.linalg.norm(pmm1 - p0)
    n0m2 = (pmm2 - p0) / np.linalg.norm(pmm2 - p0)
    rot = np.array([[0,1], [-1,0]])                               # rotation matrix
    flip = np.array([[-1,0], [0,-1]])                           # flip matrix
    line = (                                                    # `line = True` if boundary is a line
            (np.all(n01<=n0m+tol) and np.all(n01>=n0m-tol))
            or
            (np.all(n01<=flip@n0m+tol) and np.all(n01>=flip@n0m-tol))
        ) and (
            (np.all(n01<=n0m1+tol) and np.all(n01>=n0m1-tol))
            or
            (np.all(n01<=flip@n0m1+tol) and np.all(n01>=flip@n0m1-tol))
        ) and (
            (np.all(n01<=n0m2+tol) and np.all(n01>=n0m2-tol))
            or
            (np.all(n01<=flip@n0m2+tol) and np.all(n01>=flip@n0m2-tol))
        )
    if line:                                                    # enters if boundary is a line
        n = rot@n01                                               # rotation of vector n01
    else:                                                       # if boundary is not a line it has curvature
        d = np.sqrt((p[i,0]-p[b,0])**2 + (p[i,1]-p[b,1])**2)    # distances to interface nodes
        imin = b[d.argsort()[:3]]                               # index of the three closest nodes
        pm = np.mean(p[imin,:], axis=0)                         # centroid of the three closest nodes
        n1 = p[imin[1]] - pm
        n2 = p[imin[2]] - pm
        n = rot @ (n2-n1)/np.linalg.norm(n2-n1)
        pm2 = np.mean(p,0)
        n = n * np.dot(n , p[i]-pm2)
        n = n/np.linalg.norm(n)
    return n