import numpy as np
import matplotlib.pyplot as plt


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

def visualize_robot(fk, obstacle: np.ndarray = None, color='blue'):
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