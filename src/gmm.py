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
    MESH_PREFIX = 'franka_description/meshes/refined/'
    BINARY_PREFIX = '.bin/'
    def __init__(self, urdf_file: str, gmm_configuration_file: str = 'config/gmm.json') -> None:
        # get neutral frame's translation and rotation
        self._model = pin.buildModelFromUrdf(urdf_file)
        self._data = self._model.createData() 
        pin.forwardKinematics(self._model, self._data, pin.neutral(self._model))
        pin.updateFramePlacements(self._model, self._data)
        # fit a gmm per link
        self._gmms = []
        with open(gmm_configuration_file) as configuration_file:
            configuration = json.load(configuration_file)
            for i in range(self._model.nq+1):
                self._gmms.append(self._get_gmm_model(link_id=i, n_components=configuration[str(i)]))

    @classmethod
    def _extract_surface_points(cls, link_id: int)->np.ndarray:
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
        assert ((rotation.ndim == 2) and (rotation.shape[0] == rotation.shape[1]))
        transformed_object = obj.copy()
        if rotation is not None:
            transformed_object = rotation @ transformed_object @ rotation.T
        if translation is not None:
            transformed_object += translation
        return transformed_object

    def _rebase_surface(self, link_id: int):
        frame_name = f'panda_link{link_id}'
        frame_id = self._model.getFrameId(frame_name)
        rotation = self._data.oMf[frame_id].rotation
        surface = self._extract_surface_points(link_id)
        return np.dot(surface, rotation.T)

    def _get_gmm_model(self, link_id: int, n_components: int)->GaussianMixture:
        model_path = self.BINARY_PREFIX + f'link{link_id}'
        if Path(model_path).is_file():
            with open(model_path, 'rb') as file:
                gmm = pickle.load(file)
                if gmm.n_components == n_components:
                    return gmm
                else:
                    print(f'refitting GMM of link{link_id} with {n_components} components...')
        global_surface = self._rebase_surface(link_id)
        link_gmm = self._fit_gmm(global_surface, n_components=n_components)
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