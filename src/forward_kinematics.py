import time
from typing import List
import numpy as np
import pinocchio as pin
from .urdf_parser import URDFParser


class ForwardKinematic:
    def __init__(self, urdf_file: str, dim: int = 3, components_per_link: int = 1) -> None:
        parser = URDFParser(urdf_file=urdf_file, components_per_link=components_per_link)
        self.links = parser.links
        self.model = pin.buildModelFromUrdf(urdf_file)
        self.data = self.model.createData()
        self.dim = dim 
        self.mus = np.zeros((self.model.nq, self.dim))
        self.sigmas = np.zeros((self.model.nq, self.dim, self.dim))
        self.dmus = np.zeros((self.model.nq, self.dim, self.model.nq))
        self.dsigmas = np.zeros((self.model.nq, self.dim, self.dim, self.model.nq))
        self.ddmus = np.zeros((self.model.nq, self.model.nq, self.model.nq, self.dim))
        self.ddsigmas = np.zeros((self.model.nq, self.model.nq, self.model.nq, self.dim, self.dim))

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

        for i in range(self.model.nq):
            link_name = f'link{i+1}' if 'planar' in self.model.name else f'panda_link{i}'
            link_id = self.model.getFrameId(link_name)
            rotation = self.data.oMf[link_id].rotation
            link_rotation = np.linalg.inv(rotations[-1]) @ rotation if i > 0 else rotation
            rotations.append(rotation)
            link_rotations.append(link_rotation)
            translation = self.data.oMf[link_id].translation
            # computing the new mu and sigma
            self.sigmas[i] = rotation @ self.links[i].covs @ rotation.transpose(1, 0)
            # offseted_covs = self.rotation_offsets[i] @ self.links[i].covs @ self.rotation_offsets[i].T
            # self.sigmas[i] = rotation @ offseted_covs @ rotation.transpose(1, 0)
            self.mus[i] = translation + (rotation @ self.links[i].means).reshape(translation.shape)
            # self.mus[i] = np.dot(translation + (rotation @ self.links[i].means).reshape(translation.shape), self.rotation_offsets[i].T)
            # computing the first order derivative of mu and sigma
            if derivation_order > 0:
                Js.append(pin.computeFrameJacobian(self.model, self.data, q, link_id))
                dR_dqs.append(self.skew_matrix(Js[-1][3:, i]) @ link_rotation)
                dR = self.rotation_derivative(link_rotations, dR_dqs)
                self.dmus[i] = Js[-1][:3] + (dR.transpose(2, 0, 1) @ self.links[i].means).squeeze().T
                self.dsigmas[i] = rotation @ self.links[i].covs @ dR + (rotation @ self.links[i].covs @ dR).transpose(1, 0, 2)
            # computing the second order derivative of mu and sigma
            if derivation_order > 1:
                ddR_ddqs.append(self.skew_matrix(Js[-1][3:, i]) @ dR_dqs[-1])
                ddR = self.rotation_hessian(link_rotations, dR_dqs, ddR_ddqs)
                self.ddmus[i] = self.second_derivative_mu(i, link_id, ddR)
                self.ddsigmas[i] = self.second_derivative_sigma(i, link_rotation, dR, ddR)
            
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

    def second_derivative_sigma(self, joint_id, rotation, gradient, hessian):
        a = np.empty((self.model.nq, self.model.nq, self.dim, self.dim))
        b = np.empty_like(a)
        c = np.empty_like(b)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                a[i, j] = hessian[i, j] @ self.links[joint_id].covs @ rotation.T
                b[i, j] = 2 * gradient[:, :, i] @ self.links[joint_id].covs @ gradient[:, :, j].T
                c[i, j] = rotation @ self.links[joint_id].covs @ hessian[i, j].T
        return a + b + c

    def second_derivative_mu(self, joint_id, link_index, hessian):
        ddmu = np.zeros((self.model.nq, self.model.nq, self.dim, 1))
        dda = pin.getFrameAccelerationDerivatives(self.model, self.data, link_index, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[3]
        ddmu[0, 0] = np.array([[-1], [1], [1]]) * dda[:3, 0][:, np.newaxis][[1, 0, 2]]
        for i in range(1, joint_id):
            ddmu[i, i] = np.array([[-1], [1], [1]]) * dda[:3, i][:, np.newaxis][[1, 0, 2]]
            ddmu[i, i-1] = np.array([[-1], [1], [1]]) * dda[:3, i][:, np.newaxis][[1, 0, 2]]
            ddmu[i-1, i] = np.array([[-1], [1], [1]]) * dda[:3, i][:, np.newaxis][[1, 0, 2]]
        ddmu += hessian @ self.links[joint_id].means
        return ddmu.squeeze()
