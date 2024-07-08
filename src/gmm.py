from pathlib import Path
from typing import Tuple
import json
import pickle
import numpy as np
import pinocchio as pin
import stl.mesh as meshlib
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from utils.visualization import plot_3d_ellipsoid_from_covariance


class RobotModel:
    MESH_PREFIX = '../description/franka_description/meshes/visual/'
    BINARY_PREFIX = '../.bin/'
    def __init__(self, urdf_file: str, gmm_configuration_file: str) -> None:
        """Fits at least one GMM per link

        Args:
            urdf_file (str): the urdf description of the manipulator
            gmm_configuration_file (str): number of gmms per link
        """
        # get neutral frame's translation and rotation
        self._model = pin.buildModelFromUrdf(urdf_file)
        self._data = self._model.createData() 
        pin.forwardKinematics(self._model, self._data, pin.neutral(self._model))
        pin.updateFramePlacements(self._model, self._data)
        # fit a gmm per link
        self._gmms = []
        self.n_components = 0
        self.partitions = []
        with open(gmm_configuration_file) as configuration_file:
            configuration = json.load(configuration_file)
            for i in range(self._model.nq):
                self.n_components += configuration[str(i)]
                self.partitions.append(self.n_components)
                if self._model.name == 'panda':
                    gmm = self._get_gmm_model(link_id=i, n_components=configuration[str(i)])
                else:
                    # if no stl are available, fit a gmm on a fictive link
                    gmm = Link(n_components=configuration[str(i)]).gmm
                self._gmms.append(gmm)

    @classmethod
    def _extract_surface_points(cls, link_id: int)->np.ndarray:
        '''create the pointcloud representation of an stl file'''
        # select one points from each vertices
        mesh_name = cls.MESH_PREFIX + f'visual_link{link_id}.stl'
        return meshlib.Mesh.from_file(mesh_name).points[:, :3]

    @staticmethod
    def _fit_gmm(points: np.ndarray, n_components: int)->GaussianMixture:
        gmm = GaussianMixture(n_components=n_components, covariance_type='full')
        gmm.fit(points)
        return gmm

    @staticmethod
    def _to_global(obj: np.ndarray, rotation: np.ndarray, translation: np.ndarray = None)->np.ndarray:
        '''convert coordinates relative to an object to global coordinates'''
        assert ((rotation.ndim == 2) and (rotation.shape[0] == rotation.shape[1]))
        transformed_object = obj.copy()
        if rotation is not None:
            transformed_object = rotation @ transformed_object @ rotation.T
        if translation is not None:
            transformed_object += translation
        return transformed_object

    def _rebase_surface(self, link_id: int):
        '''rotate a link to match proper orientation'''
        frame_name = f'panda_link{link_id}'
        frame_id = self._model.getFrameId(frame_name)
        rotation = self._data.oMf[frame_id].rotation
        surface = self._extract_surface_points(link_id)
        return np.dot(surface, rotation.T)

    def _get_gmm_model(self, link_id: int, n_components: int)->GaussianMixture:
        '''If a gmm model with the right number of components is alread existing,
        fetch it. Otherwise recreate a new gmm model'''
        model_path = self.BINARY_PREFIX + f'link{link_id}'
        if Path(model_path).is_file():
            with open(model_path, 'rb') as file:
                gmm = pickle.load(file)
                if gmm.n_components == n_components:
                    return gmm
                else:
                    print(f'refitting GMM of link{link_id} with {n_components} components...')
        surface = self._extract_surface_points(link_id)
        link_gmm = self._fit_gmm(surface, n_components=n_components)
        with open(model_path, 'wb') as file:
            pickle.dump(link_gmm, file)
        return link_gmm

    def export_link(self, link_id: int)->Tuple[np.ndarray]:
        gmm = self._gmms[link_id]
        means = gmm.means_
        priors = gmm.weights_
        covariances = gmm.covariances_
        return means, covariances, priors
    
    def display_link(self, link_id: int):
        s = self._rebase_surface(link_id)
        m, c, p =  self.export_link(link_id)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(s[:, 0], s[:, 1], s[:, 2], s=1, alpha=0.01)
        ax.scatter(m[:, 0], m[:, 1], m[:, 2], s=100, c='yellow', label='means')
        for i in range(c.shape[0]):
            ax.set_aspect('equal')
            plot_3d_ellipsoid_from_covariance(c[i], center=m[i], ax=ax, color='orange')
        return ax

    @property
    def gmms(self):
        return self._gmms


class Link:
    """Create a fictive link with an ellispoid shape
    """
    def __init__(self, n_components) -> None:
        points = self._generate_ellipsoid().T[[0, 2, 1]].T
        self.points = points.copy()
        self.gmm = GaussianMixture(n_components=n_components)
        self.gmm.fit(self.points)

    @staticmethod
    def _generate_ellipsoid(a=0.25, b=0.25, c=0.5, num_points=1000):
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