from pathlib import Path
from typing import Tuple
import json
import pickle
import numpy as np
import pinocchio as pin
import stl.mesh as meshlib
from sklearn.mixture import GaussianMixture


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
    def _to_global(obj: np.ndarray, rotation: np.ndarray, translation: np.ndarray)->np.ndarray:
        assert ((rotation.ndim == 2) and (rotation.shape[0] == rotation.shape[1]))
        return rotation @ obj @ rotation.T + translation

    def _get_gmm_model(self, link_id: int, n_components: int)->GaussianMixture:
        model_path = self.BINARY_PREFIX + f'link{link_id}'
        if Path(model_path).is_file():
            with open(model_path, 'rb') as file:
                gmm = pickle.load(file)
                if gmm.n_components == n_components:
                    return gmm
                else:
                    print(f'refitting GMM of link{link_id} with {n_components} components...')
        surface_points = self._extract_surface_points(link_id)
        link_gmm = self._fit_gmm(surface_points, n_components=n_components)
        with open(model_path, 'wb') as file:
            pickle.dump(link_gmm, file)
        return link_gmm

    def export_link(self, link_id: int)->Tuple[np.ndarray]:
        gmm = self._gmms[link_id]
        frame_name = f'panda_link{link_id}'
        frame_id = self._model.getFrameId(frame_name)
        translation = self._data.oMf[frame_id].translation
        rotation = self._data.oMf[frame_id].rotation
        means = gmm.means_
        priors = gmm.weights_
        covariances = self._to_global(gmm.covariances_, rotation=rotation, translation=translation)
        return means, covariances, priors
