import time
from typing import List
import numpy as np
import pinocchio as pin
from .urdf_parser import URDFParser
from .gmm import RobotModel


class ForwardKinematic:
    def __init__(self, urdf_file: str, gmm_configuration_file: str, dim: int = 3) -> None:
        self.robot_model = RobotModel(urdf_file, gmm_configuration_file=gmm_configuration_file)
        self.model = pin.buildModelFromUrdf(urdf_file)
        self.data = self.model.createData()
        self.dim = dim 
        n_gmms = self.robot_model.n_components
        self.priors = np.hstack([gmm.weights_ for gmm in self.robot_model.gmms])
        self.mus = np.zeros((n_gmms, self.dim))
        self.sigmas = np.zeros((n_gmms, self.dim, self.dim))
        self.dmus = np.zeros((n_gmms, self.dim, self.model.nq))
        self.dsigmas = np.zeros((n_gmms, self.dim, self.dim, self.model.nq))
        self.ddmus = np.zeros((n_gmms, self.model.nq, self.model.nq, self.dim))
        self.ddsigmas = np.zeros((n_gmms, self.model.nq, self.model.nq, self.dim, self.dim))


    def profiler(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            print(f'execution frequency of {func.__name__}: {1/(time.time() - start):.4f} Hz')
            return res
        return wrapper

    def __call__(self, q: np.ndarray, dq: np.ndarray, derivation_order: int = 2):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        pin.computeForwardKinematicsDerivatives(self.model, self.data, q, dq, np.zeros_like(dq))
        rotations = []
        link_rotations = []
        Js = []
        dR_dqs = []
        ddR_ddqs = []
        k = 0
        for i in range(self.model.nq):
            means, covs, _ = self.robot_model.export_link(i)
            link_name = f'link{i+1}' if 'planar' in self.model.name else f'panda_link{i}'
            link_id = self.model.getFrameId(link_name)
            rotation = self.data.oMf[link_id].rotation
            link_rotation = np.linalg.inv(rotations[-1]) @ rotation if i > 0 else rotation
            rotations.append(rotation)
            link_rotations.append(link_rotation)
            translation = self.data.oMf[link_id].translation
            Js.append(pin.computeFrameJacobian(self.model, self.data, q, link_id))
            dR_dqs.append(self.skew_matrix(Js[-1][3:, i]) @ link_rotation)
            dR = self.rotation_derivative(link_rotations, dR_dqs)
            ddR_ddqs.append(self.skew_matrix(Js[-1][3:, i]) @ dR_dqs[-1])
            ddR = self.rotation_hessian(link_rotations, dR_dqs, ddR_ddqs)
            dJ = self.jacobian_derivative(i, link_id)
            for j in range(means.shape[0]):
                self.sigmas[k] = rotation @ covs[j] @ rotation.transpose(1, 0)
                self.mus[k] = translation + rotation @ means[j]
                if derivation_order > 0:
                    self.dmus[k] = Js[-1][:3] + (dR.transpose(2, 0, 1) @ means[j]).squeeze().T
                    self.dsigmas[k] = rotation @ covs[j] @ dR + (rotation @ covs[j] @ dR).transpose(1, 0, 2)
                if derivation_order > 1:
                    self.ddmus[k] = self.second_derivative_mu(means[j], ddR, dJ)
                    self.ddsigmas[k] = self.second_derivative_sigma(covs[j], link_rotation, dR, ddR)
                k+=1
        return self.mus, self.sigmas, self.dmus, self.dsigmas, self.ddmus, self.ddsigmas


    @staticmethod
    def skew_matrix(axis: np.ndarray):
        return np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
    
    def rotation_derivative(self, rotations: List[np.ndarray], drotations: List[np.ndarray]):
        dR = np.zeros((self.model.nq, self.dim, self.dim))
        for i in range(len(rotations)):
            r = np.eye(rotations[i].shape[0])
            for j in range(len(rotations)):
                if i == j: r @= drotations[i]
                else: r @= rotations[j]
            dR[i] += r
        return dR.transpose(1, 2, 0)
    
    def rotation_hessian(self, rotations: List[np.ndarray], drotations: List[np.ndarray], ddrotations: List[np.ndarray]):
        ddR = np.zeros((self.model.nq, self.model.nq, self.dim, self.dim))
        for i in range(len(rotations)):
            for j in range(len(rotations)):
                ddr = np.eye(rotations[0].shape[0])
                if i == j:
                    for n in range(len(rotations)-1, -1, -1):
                        if i == n:
                            ddr @= ddrotations[n]
                        else:
                            ddr @= rotations[n]
                else:
                    for n in range(len(rotations)-1, -1, -1):
                        if (i == n) or (j == n):
                            ddr @= drotations[n]
                        else:
                            ddr @= rotations[n]
                ddR[i, j] = ddr
        return ddR

    def second_derivative_sigma(self, cov, rotation, gradient, hessian):
        a = np.empty((self.model.nq, self.model.nq, self.dim, self.dim))
        b = np.empty_like(a)
        c = np.empty_like(b)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                a[i, j] = hessian[i, j] @ cov @ rotation.T
                b[i, j] = 2 * gradient[:, :, i] @ cov @ gradient[:, :, j].T
                c[i, j] = rotation @ cov @ hessian[i, j].T
        return a + b + c

    def second_derivative_mu(self, mean, hessian, dJ):
        ddmu = dJ + hessian @ mean
        return ddmu.squeeze()

    def jacobian_derivative(self, link_index, link_id):
        ddmu = np.zeros((self.model.nq, self.model.nq, self.dim))
        dda = pin.getFrameAccelerationDerivatives(self.model, self.data, link_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[3]
        ddmu[0, 0] = np.array([-1, 1, 1]) * dda[:3, 0][[1, 0, 2]]
        for i in range(1, link_index):
            ddmu[i, i] = np.array([-1, 1, 1]) * dda[:3, i][[1, 0, 2]]
            ddmu[i, i-1] = np.array([-1, 1, 1]) * dda[:3, i][[1, 0, 2]]
            ddmu[i-1, i] = np.array([-1, 1, 1]) * dda[:3, i][[1, 0, 2]]
        return ddmu