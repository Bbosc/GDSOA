from itertools import combinations
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_3d_ellipsoid_from_covariance(covariance_matrix, center=[0, 0, 0], ax=None, color='blue'):
    # From chatgpt
    """
    Plot a 3D ellipsoid from a covariance matrix.

    Args:
    covariance_matrix (array-like): Covariance matrix of shape (3, 3).
    center (array-like, optional): Center of the ellipsoid. Default is [0, 0, 0].
    ax (matplotlib.axes.Axes, optional): Axes to plot on. If not provided, a new figure will be created.

    Returns:
    matplotlib.axes.Axes: Axes object containing the plot.
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # Eigenvalue decomposition of covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

    # Sort eigenvalues and corresponding eigenvectors
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Radii of the ellipsoid (square root of eigenvalues)
    radii = np.sqrt(eigenvalues)
    # radii = np.sqrt(np.diag(covariance_matrix))

    # Create sphere mesh
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 10)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones(np.size(u)), np.cos(v))

    # Apply rotation
    points = np.array([x.flatten(), y.flatten(), z.flatten()]).T
    rotated_points = np.dot(points, eigenvectors.T)
    x = np.reshape(rotated_points[:, 0], x.shape)
    y = np.reshape(rotated_points[:, 1], y.shape)
    z = np.reshape(rotated_points[:, 2], z.shape)

    # Plot ellipsoid
    ax.plot_surface(x + center[0], y + center[1], z + center[2], color=color, alpha=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    return ax

def visualize_robot(fk, obstacle: np.ndarray = None, color='blue', ax = None):
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
    ax.scatter(*(fk.mus[:, i] for i in range(3)), c=color)
    for i in range(fk.sigmas.shape[0]):
        plot_3d_ellipsoid_from_covariance(fk.sigmas[i], center=fk.mus[i], ax=ax, color=color)
    if obstacle is not None:
        ax.scatter(obstacle[:, 0], obstacle[:, 1], obstacle[:, 2], c='black', s=100)
    ax.set_xlim([-2.1, 2.1])
    ax.set_ylim([-0.1, 2.1])
    ax.set_zlim([-0.1, 1.3])
    ax.axis('equal')
    return ax

def generate_grid_coordinates(joint_limits: List[dict], resolution: int=5):
    qs = tuple(np.linspace(limit['lower'], limit['upper'], resolution) for limit in joint_limits)
    grids = np.meshgrid(*qs)
    return np.column_stack(tuple(g.ravel() for g in grids))

def plot_coupled_embeddings(coordinates, attractor, streamlines, embedding, start):
    couples = list(combinations(np.linspace(0, coordinates.shape[1]-1, coordinates.shape[1]), r=2))
    fig, axs = plt.subplots(coordinates.shape[1], int(len(couples)/coordinates.shape[1]))
    fig.set_size_inches(15, 20)
    row = -1
    for b, couple in enumerate(couples):
        if axs.ndim == 1:
            ax = axs[b]
        else:
            if b%axs.shape[1] == 0: row +=1
            ax = axs[row, b%axs.shape[1]]
        angle1 = min(int(couple[0]), int(couple[1]))
        angle2 = max(int(couple[0]), int(couple[1]))
        x = np.unique(embedding[:, angle1])
        y = np.unique(embedding[:, angle2])
        f = embedding[:, -1].reshape(tuple(x.shape[0] for _ in range(embedding.shape[1]-1)))
        dims_to_sum = tuple(map(lambda tup: tup[0], filter(lambda tup: tup[1], [(i, i not in (angle1, angle2)) for i in range(coordinates.shape[1])])))
        z = f.sum(dims_to_sum) 
        if angle1 == 0:
            ax.contourf(x, y, z, antialiased=False, alpha=0.35, cmap=cm.coolwarm, levels=10)
        else:
            ax.contourf(x, y, z.T, antialiased=False, alpha=0.35, cmap=cm.coolwarm, levels=10)
        if attractor is not None:
            ax.scatter(attractor[angle1], attractor[angle2], marker='*', label='target', c='navy', s=80)
        if start is not None:
            ax.scatter(start[angle1], start[angle2], marker='*', label='start', c='gold', s=80)
        if streamlines is not None:
            ax.scatter(streamlines[:, angle1], streamlines[:, angle2], label='path', c='black', s=1)
        ax.set_xlabel(f'q{angle1+1}')
        ax.set_ylabel(f'q{angle2+1}')
        ax.set_xlim([x.min(), x.max()])
        ax.set_ylim([y.min(), y.max()])
        ax.legend(loc='upper right')
    fig.tight_layout()
    return ax