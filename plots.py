"""Ploting"""

import numpy as np
import matplotlib.pyplot as plt

def plot_nodes(p, b, labels=(), figsize=(8,4), size=150, nums=False, alpha=0.75):
    """Scatter plot for different arrays of index nodes `b`
    
    Parameters
    -----
    p : array_like
        Node coordinates, ordered by rows p[0] = np.array([x0, y0]).
    b : sequence of ndarrays
        Index nodes arrays to plot.
    labels : sequence of strings
        Labels of the different arrays.
    size : integer or real
        Size of each node in the scatter plot.
    num : boolean
        If `True` then text with the number of node is included.
    """
    fig = plt.figure(figsize=figsize)
    i = 1
    for bi in b:
        if i > len(labels):
            label = ''
        else:
            label = labels[i-1]
        plt.scatter(p[bi,0], p[bi,1], label=label, s=size, alpha=alpha)
        i += 1
    if nums:
        for i in np.hstack(b):
            plt.text(p[i,0], p[i,1], str(i))
    plt.axis('equal')
    plt.legend()
    plt.title('Nodes')
    return fig

def plot_normal_vectors(b,p):
    fig = plt.figure()
    plt.scatter(p[b,0], p[b,1], s=70)
    from GFDMI import normal_vectors
    n = normal_vectors(b,p)
    plt.quiver(p[b,0], p[b,1], n[:,0], n[:,1], alpha=0.5)
    plt.axis("equal")
    return fig

def tri_surface(p, t, U, azim=-60, elev=30):
    """3D surface plot.
    
    Parameters
    -----
    p : array_like
        Node coordinates, ordered by rows p[0] = np.array([x0, y0]).
    t : array_like
        Triangulations.
    U : ndarray
        Values in each node.
    """
    fig = plt.figure(figsize=(7,7))
    ax = plt.axes(projection='3d')
    ax.plot_trisurf(p[:,0], p[:,1], U,
                cmap=plt.get_cmap('cool'),
                edgecolor=(0,0,0,0.2),
                alpha=0.75)
    ax.view_init(azim=azim, elev=elev)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('U')
    return fig, ax

def contourf_plot(p, U, levels=20):
    """Countourf of values `U`
    
    Parameters
    -----
    p : array_like
        Node coordinates, ordered by rows p[0] = np.array([x0, y0]).
    U : ndarray
        Values in each node.
    """
    fig = plt.figure(figsize=(7,4))
    plt.tricontour(p[:,0], p[:,1], U, levels, colors="k", linewidths=0.5)
    plt.tricontourf(p[:,0], p[:,1], U, levels)
    plt.colorbar()
    plt.axis('equal')
    plt.xlabel('x')
    plt.ylabel('y')
    return fig