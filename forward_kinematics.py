import time
from typing import List
import numpy as np
import pinocchio as pin
from urdf_parser import URDFParser


class ForwardKinematic:
    def __init__(self, urdf_file: str) -> None:
        parser = URDFParser(urdf_file=urdf_file)
        self.links = parser.links
        self.model = pin.buildModelFromUrdf(urdf_file)
        self.data = self.model.createData()

    
    def __call__(self, q: np.ndarray, derivation_order: int = 0):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)

        rotations = []
        Js = []
        dR_dqs = []

        for i in range(self.model.nq):
            link_id = self.model.getFrameId(f'link{i+1}')
            rotation = self.data.oMf[link_id].rotation
            rotations.append(rotation)
            translation = self.data.oMf[link_id].translation
            # computing the new mu and sigma
            covariance = rotation @ self.links[i].covs @ rotation.transpose(1, 0)
            mu = translation + rotation @ self.links[i].means
            # computing the first order derivative of mu and sigma
            if derivation_order > 0:
                Js.append(pin.computeFrameJacobian(self.model, self.data, q, link_id))
                dR_dqs.append(self.skew_matrix(Js[-1][3:, i]) @ rotation)
                dR = self.rotation_derivative(rotations, dR_dqs)
                dmu = Js[-1][:3] + dR @ self.links[i].means
                dsigma = rotation @ self.links[i].covs @ dR + (rotation @ self.links[i].covs @ dR).transpose(1, 0, 2)
            # computing the second order derivative of mu and sigma
            if derivation_order > 1:
                pass

    @staticmethod
    def skew_matrix(axis: np.ndarray):
        return np.array([
            [0, -axis[2], axis[1]],
            [axis[2], 0, -axis[0]],
            [-axis[1], axis[0], 0]
        ])
    
    def rotation_derivative(self, rotations: List[np.ndarray], drotations: List[np.ndarray]):
        dR = np.zeros((drotations[0].shape))
        r = np.zeros(rotations[i].shape)
        for i in range(len(rotations)):
            dR += r
            r = np.eye(rotations[i].shape[0])
            for j in range(len(rotations)):
                if i == j: r @= drotations[i]
                else: r @= rotations[i]
        return dR
    
    def rotation_hessian(self, rotations: List[np.ndarray], drotations: List[np.ndarray], ddrotations: List[np.ndarray]):
        ddR = np.zeros((self.model.nq, self.model.nq, rotations[0].shape[0], rotations[0].shape[1]))
        for i in range(ddR.shape[0]):
            for j in range(ddR.shape[0]):
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



    def test(self, q: np.ndarray):
        pin.forwardKinematics(self.model, self.data, q)
        pin.updateFramePlacements(self.model, self.data)
        link_global_rotations = np.array([self.data.oMf[i].rotation for i in self.link_ids])
        link_global_translations = np.array([self.data.oMf[i].translation for i in self.link_ids])
        local_rotations = [link_global_rotations[0]]
        for i in range(1, len(link_global_rotations)):
            local_rotations.append(np.linalg.inv(link_global_rotations[i-1]) @ link_global_rotations[i])
        link_local_rotations = np.array(local_rotations)
        drotations = self.build_rotation_derivative_exp(link_local_rotations)
        ddrotations = self.build_rotation_second_derivative_exp(link_local_rotations)
        
        sigmas = link_global_rotations @ self.initial_sigma @ link_global_rotations.transpose(0, 2, 1)
        mus = link_global_translations + np.einsum('bij,bj->bi', link_global_rotations, self.initial_mu.squeeze())
        rotation_gradients = []
        dsigma = np.zeros((self.model.nq, sigmas.shape[1], sigmas.shape[2], self.model.nq))
        dmu = np.zeros((self.model.nq, mus.shape[1], self.model.nq))
        ddsigma = np.zeros((self.model.nq, self.model.nq, self.model.nq, sigmas.shape[1], sigmas.shape[2]))
        ddmu = np.zeros((self.model.nq, self.model.nq, self.model.nq, mus.shape[1]))
        for i in range(self.model.nq):
            # computing the rotation gradient
            # rotation_gradient = self.rotation_gradient(local_rotations=link_local_rotations[:i+1], local_rotations_derivative=drotations[:i+1])
            rotation_gradient = self.rotation_gradient(i, link_local_rotations, drotations).transpose(1, 2, 0)
            rotation_gradients.append(rotation_gradient)
            # computing the first order dervative
            # dsigma[i] = (rotation_gradient @ self.initial_sigma[i] @ link_global_rotations[i].T).T + link_global_rotations[i] @ self.initial_sigma[i] @ rotation_gradient.T
            dsigma[i] = link_global_rotations[i] @ self.initial_sigma[i] @ rotation_gradient + (link_global_rotations[i] @ self.initial_sigma[i] @ rotation_gradient).transpose(1, 0, 2)
            if i == 0:
                dmu[i] = (rotation_gradient.transpose(2, 0, 1) @ self.initial_mu[i]).squeeze().T
            else:
                dmu[i] = (rotation_gradient.transpose(2, 0, 1) @ self.initial_mu[i] + rotation_gradients[i-1].transpose(2, 0, 1) @ self.initial_t[i-1]).squeeze().T
            # computing the rotation hessian
            rotation_hessian = self.rotation_hessian(i, link_local_rotations, drotations, ddrotations)
            # computing the second order derivative
            ddsigma[i] = self.second_derivative_sigma(self.initial_sigma[i], link_global_rotations[i], rotation_gradient.transpose(2, 0, 1), rotation_hessian)
            ddmu[i] = self.second_derivative_mu(i, rotations=link_local_rotations, drotations=drotations, ddrotations=ddrotations)
        return mus, sigmas, dmu, dsigma, ddmu, ddsigma

        
    @staticmethod
    def build_rotation_derivative_exp(rotations):
        dR = np.zeros_like(rotations)
        dR[:, 0, 0] = rotations[:, 0, 1]
        dR[:, 1, 1] = rotations[:, 0, 1]
        dR[:, -1, -1] = 0.
        dR[:, 1, 0] = rotations[:, 0, 0]
        dR[:, 0, 1] = -rotations[:, 0, 0]
        return dR

    @staticmethod
    def build_rotation_second_derivative_exp(rotations):
        dR = - rotations.copy()
        dR[:, -1, -1] = 0.
        return dR
    
    def rotation_gradient_n(self, local_rotations, local_rotations_derivative):
        gradient = np.zeros((self.model.nq, local_rotations.shape[1], local_rotations.shape[2]))
        for i in range(local_rotations.shape[0]):
            rd = np.eye(local_rotations[i].shape[0])
            if local_rotations[:i].shape[0] == 1: 
                rd = local_rotations[:i]
            elif local_rotations[:i].shape[0] > 1:
                rd = np.linalg.multi_dot(local_rotations[:i])
            ru = np.eye(local_rotations[i].shape[0])
            if local_rotations[i+1:].shape[0] == 1: 
                ru = local_rotations[i+1:]
            elif local_rotations[i+1:].shape[0] > 1:
                ru = np.linalg.multi_dot(local_rotations[i+1:])
            gradient[i] = rd @ local_rotations_derivative[i] @ ru
        return gradient
    
    def rotation_gradient(self, link_index, rotations, drotations):
        gradient = np.zeros((self.model.nq, rotations.shape[1], rotations.shape[1]))
        if link_index == 1:
            gradient[0] = drotations[0] @ rotations[1]
            gradient[1] = rotations[0] @ drotations[1]
        elif link_index == 0:
            gradient[0] = drotations[0]
        else:
            raise NotImplementedError
        return gradient

    def rotation_hessian(self, link_index, rotations, drotations, ddrotations):
        hessian = np.zeros((self.model.nq, self.model.nq, rotations.shape[1], rotations.shape[1]))
        if link_index == 1:
            hessian[0, 0] = ddrotations[0] @ rotations[1]
            hessian[1, 1] = rotations[0] @ ddrotations[1]
            hessian[0, 1] = drotations[0] @ drotations[1]
            hessian[1, 0] = drotations[0] @ drotations[1]
        elif link_index == 0:
            hessian[0, 0] = ddrotations[0]
        else:
            raise NotImplementedError
        return hessian


    @staticmethod
    def build_rotation_derivative(rotation):
        dR = np.zeros_like(rotation)
        np.fill_diagonal(dR, rotation[0, 1])
        dR[-1, -1] = 0.
        dR[1, 0] = rotation[0, 0]
        dR[0, 1] = -rotation[0, 0]
        return dR
    
    @staticmethod
    def build_intermediate_rotations(angles):
        rotations = np.zeros((angles.shape[0], 3, 3))
        for i, angle in enumerate(angles):
            c, s = np.cos(angle), np.sin(angle)
            rotations[i] = np.array([
                [c, -s, 0],
                [s, c, 0],
                [0, 0, 1]
            ])
        return rotations
    
    def derive_rotation(self, angles, rotations):
        dR = np.zeros((angles.shape[0], rotations.shape[1], rotations.shape[2]))
        for i in range(len(angles)):
            drotation = self.build_rotation_derivative(rotations[i])
            rm = rotations[:i] if rotations[:i].size != 0 else np.eye(rotations.shape[1])
            rp = rotations[i+1:angles.shape[0]] if rotations[i+1:angles.shape[0]].size != 0 else np.eye(rotations.shape[1])
            dR[i] = rm @ drotation @ rp
        return dR
    
    @staticmethod
    def second_derivative_sigma(sigma_ref, rotation, gradient, hessian):
        m = np.zeros((gradient.shape[0], gradient.shape[0], rotation.shape[1], rotation.shape[1]))
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

