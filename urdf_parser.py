import numpy as np
import pinocchio as pin
from sklearn.mixture import GaussianMixture


class URDFParser:
    def __init__(self, urdf_file: str) -> None:
        self.model = pin.buildModelFromUrdf(urdf_file)
        self.data = self.model.createData()
        self.links = [Link() for _ in range(self.model.nq)]


class Link:
    def __init__(self, stl_file: str = None, n_components: int = 1) -> None:
        if stl_file is None:
            points = generate_ellipsoid().T[[0, 2, 1]].T
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(points)
        self.means: np.ndarray = gmm.means_.transpose(1, 0)
        self.priors: np.ndarray = gmm.weights_
        self.covs: np.ndarray = gmm.covariances_
        self.vector: np.ndarray = points[np.argmax(points[:, 2])] - points[np.argmin(points[:, 2])]


def generate_ellipsoid(a=0.25, b=0.25, c=0.5, num_points=1000):
    """
    Generates a 3D point cloud representing an ellipsoid.

    Parameters:
    a (float): Semi-axis length along the x-axis.
    b (float): Semi-axis length along the y-axis.
    c (float): Semi-axis length along the z-axis.
    num_points (int): Number of points in the point cloud.

    Returns:
    np.ndarray: Array of shape (num_points, 3) representing the point cloud.
    """
    # Generate random points on a unit sphere
    phi = np.random.uniform(0, 2 * np.pi, num_points)
    cos_theta = np.random.uniform(-1, 1, num_points)
    sin_theta = np.sqrt(1 - cos_theta**2)
    
    x = sin_theta * np.cos(phi)
    y = sin_theta * np.sin(phi)
    z = cos_theta
    
    # Scale points to the ellipsoid
    x *= a
    y *= b
    z *= c
    
    return np.vstack((x, y, z + c)).T