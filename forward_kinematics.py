import time
from typing import List
import numpy as np
import pinocchio as pin
from urdf_parser import URDFParser


class ForwardKinematic:
    def __init__(self, urdf_file: str, dim: int = 3) -> None:
        parser = URDFParser(urdf_file=urdf_file)
        self.links = parser.links
        self.model = pin.buildModelFromUrdf(urdf_file)
        self.data = self.model.createData()
        self.dim = dim 

    
    def __call__(self, q: np.ndarray, derivation_order: int = 0):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        rotations = []
        link_rotations = []
        Js = []
        dR_dqs = []
        ddR_ddqs = []

        for i in range(self.model.nq):
            link_id = self.model.getFrameId(f'link{i+1}')
            rotation = self.data.oMf[link_id].rotation
            link_rotation = np.linalg.inv(rotations[-1]) @ rotation if i > 0 else rotation
            rotations.append(rotation)
            link_rotations.append(link_rotation)
            translation = self.data.oMf[link_id].translation
            # computing the new mu and sigma
            covariance = rotation @ self.links[i].covs @ rotation.transpose(1, 0)
            mu = translation + rotation @ self.links[i].means
            # computing the first order derivative of mu and sigma
            if derivation_order > 0:
                Js.append(pin.computeFrameJacobian(self.model, self.data, q, link_id))
                dR_dqs.append(self.skew_matrix(Js[-1][3:, i]) @ link_rotation)
                dR = self.rotation_derivative(link_rotations, dR_dqs)
                dmu = Js[-1][:3] + (dR.transpose(2, 0, 1) @ self.links[i].means).squeeze().T
                dsigma = rotation @ self.links[i].covs @ dR + (rotation @ self.links[i].covs @ dR).transpose(1, 0, 2)
            # computing the second order derivative of mu and sigma
            if derivation_order > 1:
                ddR_ddqs.append(self.skew_matrix(Js[-1][3:, i]) @ dR_dqs[-1])
                ddR = self.rotation_hessian(link_rotations, dR_dqs, ddR_ddqs)
                ddsigma = self.second_derivative_sigma(self.links[i].covs, link_rotation, dR, ddR)

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

    @staticmethod
    def second_derivative_sigma(sigma_ref, rotation, gradient, hessian):
        a = np.empty((gradient.shape[0], gradient.shape[0], rotation.shape[0], rotation.shape[1]))
        b = np.empty_like(a)
        c = np.empty_like(b)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                a[i, j] = hessian[i, j] @ sigma_ref @ rotation.T
                b[i, j] = 2 * gradient[i] @ sigma_ref @ gradient[j].T
                c[i, j] = rotation @ sigma_ref @ hessian[i, j].T
        return a + b + c

    def second_derivative_mu(self, link_index, rotations, drotations, ddrotations):
        ddmu = np.zeros((self.model.nq, self.model.nq, self.initial_mu.shape[1], self.initial_mu.shape[2]))
        if link_index == 0:
            ddmu[0, 0] = ddrotations[0] @ self.initial_mu[link_index]
        elif link_index == 1:
            ddmu[0, 0] = ddrotations[0] @ self.initial_t[0] + ddrotations[0] @ rotations[1] @ self.initial_mu[link_index]
            ddmu[0, 1] = drotations[0] @ drotations[1] @ self.initial_mu[link_index]
            ddmu[1, 0] = drotations[0] @ drotations[1] @ self.initial_mu[link_index]
            ddmu[1, 1] = rotations[0] @ ddrotations[1] @ self.initial_mu[link_index]
        return ddmu.squeeze()

