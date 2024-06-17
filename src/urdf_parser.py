from pathlib import Path
import numpy as np
import pinocchio as pin
from stl import mesh as meshlib
from sklearn.mixture import GaussianMixture


class URDFParser:
    def __init__(self, urdf_file: str, components_per_link: int) -> None:
        self.n_components = components_per_link
        self.model = pin.buildModelFromUrdf(urdf_file)
        self.data = self.model.createData()
        config_start = pin.neutral(self.model)
        pin.forwardKinematics(self.model, self.data, config_start)
        pin.updateFramePlacements(self.model, self.data)
        if self.model.name == 'panda':
            stl_path = Path(__file__).parent.parent / 'franka_description/meshes/visual'
            links = list(filter(lambda frame: 'panda_link' in frame.name, self.model.frames))
            links_ids = [self.model.getFrameId(frame.name) for frame in links]
            self.links = [Link(stl_file=stl_path/ f'visual_link{i}.stl',
                               rotation=self.data.oMf[links_ids[i]].rotation,
                               translation=self.data.oMf[links_ids[i]].translation,
                               n_components=components_per_link) for i in range(self.model.nq)]
        else:
            self.links = [Link(n_components=1) for _ in range(self.model.nq)]


class Link:
    def __init__(self, n_components, stl_file: str = None, rotation: np.ndarray = None, translation: np.ndarray = None) -> None:
        if stl_file is None:
            points = generate_ellipsoid().T[[0, 2, 1]].T
        else:
            points = self.get_point_from_stl(stl_file)
        self.points = points.copy()
        gmm = GaussianMixture(n_components=n_components)
        gmm.fit(self.points)
        self.means: np.ndarray = gmm.means_[:, :, np.newaxis]
        self.priors: np.ndarray = gmm.weights_
        if rotation is not None:
            self.covs: np.ndarray = rotation @ gmm.covariances_ @ rotation.T
            # self.covs: np.ndarray = rotation @ np.diag(gmm.covariances_.squeeze()) @ rotation.T
        else:
            self.covs = gmm.covariances_
        # self.covs *= 4

    def get_point_from_stl(self, stl_file: str, surface_resolution: int = 1):
        mesh = meshlib.Mesh.from_file(stl_file)
        surface_points = mesh.points[:, :3]
        surfaces = np.array_split(surface_points[np.argsort(surface_points[:, 2])], surface_resolution)
        volumes = []
        for surface in surfaces:
            inside = self._fill_surface(surface, resolution=5)
            volumes.append(inside)
        return np.concatenate(volumes + [surface_points], axis=0)

    def _fill_surface(self, surface: np.ndarray, resolution: int = 10):
        x = np.linspace(min(surface[:, 0]), max(surface[:, 0]), resolution)
        y = np.linspace(min(surface[:, 1]), max(surface[:, 1]), resolution)
        z = np.linspace(min(surface[:, 2]), max(surface[:, 2]), resolution)
        grid = np.meshgrid(x, y, z)
        flat_grid = np.stack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).T
        directions = np.zeros((flat_grid.shape[0]))
        center = self._surface_center(surface)
        for i in range(flat_grid.shape[0]):
            closest_index = np.argmin(np.linalg.norm(surface - flat_grid[i], axis=1))
            diff = flat_grid[i] - surface[closest_index]
            directions[i] = np.dot(diff-center, surface[closest_index]-center)
        return flat_grid[np.where(directions<0.)]

    @staticmethod
    def _surface_center(surface):
        center_x = np.min(surface[:,0])-(np.min(surface[:,0]) - np.max(surface[:,0]))/2
        center_y = np.min(surface[:,1])-(np.min(surface[:,1]) - np.max(surface[:,1]))/2
        center_z = np.min(surface[:,2])-(np.min(surface[:,2]) - np.max(surface[:,2]))/2
        return np.array([center_x, center_y, center_z])
    

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