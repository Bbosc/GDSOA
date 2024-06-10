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
        self.n_gmms = self.model.nq * parser.n_components
        self.mus = np.zeros((self.n_gmms, self.dim))
        self.sigmas = np.zeros((self.n_gmms, self.dim, self.dim))
        self.dmus = np.zeros((self.n_gmms, self.dim, self.model.nq))
        self.dsigmas = np.zeros((self.n_gmms, self.dim, self.dim, self.model.nq))
        self.ddmus = np.zeros((self.n_gmms, self.model.nq, self.model.nq, self.dim))
        self.ddsigmas = np.zeros((self.n_gmms, self.model.nq, self.model.nq, self.dim, self.dim))

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

        for i in range(self.n_gmms):
            link_index = int(i/(self.n_gmms/self.model.nq))
            component_id = int(i%(self.n_gmms/self.model.nq))
            if component_id == 0:
                link_name = f'link{link_index+1}' if 'planar' in self.model.name else f'panda_link{link_index}'
                link_id = self.model.getFrameId(link_name)
                rotation = self.data.oMf[link_id].rotation
                link_rotation = np.linalg.inv(rotations[-1]) @ rotation if i > 0 else rotation
                rotations.append(rotation)
                link_rotations.append(link_rotation)
                translation = self.data.oMf[link_id].translation
            # computing the new mu and sigma
            self.sigmas[i] = rotation @ self.links[link_index].covs[component_id] @ rotation.transpose(1, 0)
            self.mus[i] = translation + (rotation @ self.links[link_index].means[component_id]).reshape(translation.shape)
            # computing the first order derivative of mu and sigma
            if derivation_order > 0:
                if component_id == 0:
                    Js.append(pin.computeFrameJacobian(self.model, self.data, q, link_id))
                    dR_dqs.append(self.skew_matrix(Js[-1][3:, link_index]) @ link_rotation)
                    dR = self.rotation_derivative(link_rotations, dR_dqs)
                self.dmus[i] = Js[-1][:3] + (dR.transpose(2, 0, 1) @ self.links[link_index].means[component_id]).squeeze().T
                self.dsigmas[i] = rotation @ self.links[link_index].covs[component_id] @ dR + (rotation @ self.links[link_index].covs[component_id] @ dR).transpose(1, 0, 2)
            # computing the second order derivative of mu and sigma
            if derivation_order > 1:
                if component_id == 0:
                    ddR_ddqs.append(self.skew_matrix(Js[-1][3:, link_index]) @ dR_dqs[-1])
                    ddR = self.rotation_hessian(link_rotations, dR_dqs, ddR_ddqs)
                    dJ = self.jacobian_derivative(link_index, link_id)
                self.ddmus[i] = self.second_derivative_mu(link_index, component_id, ddR, dJ)
                self.ddsigmas[i] = self.second_derivative_sigma(link_index, component_id, link_rotation, dR, ddR)
            
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

    def second_derivative_sigma(self, joint_id, component_id, rotation, gradient, hessian):
        a = np.empty((self.model.nq, self.model.nq, self.dim, self.dim))
        b = np.empty_like(a)
        c = np.empty_like(b)
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                a[i, j] = hessian[i, j] @ self.links[joint_id].covs[component_id] @ rotation.T
                b[i, j] = 2 * gradient[:, :, i] @ self.links[joint_id].covs[component_id] @ gradient[:, :, j].T
                c[i, j] = rotation @ self.links[joint_id].covs[component_id] @ hessian[i, j].T
        return a + b + c

    
    def second_derivative_mu(self, link_index, component_id, hessian, dJ):
        ddmu = dJ + hessian @ self.links[link_index].means[component_id]
        return ddmu.squeeze()

    def jacobian_derivative(self, link_index, link_id):
        ddmu = np.zeros((self.model.nq, self.model.nq, self.dim, 1))
        dda = pin.getFrameAccelerationDerivatives(self.model, self.data, link_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED)[3]
        ddmu[0, 0] = np.array([[-1], [1], [1]]) * dda[:3, 0][:, np.newaxis][[1, 0, 2]]
        for i in range(1, link_index):
            ddmu[i, i] = np.array([[-1], [1], [1]]) * dda[:3, i][:, np.newaxis][[1, 0, 2]]
            ddmu[i, i-1] = np.array([[-1], [1], [1]]) * dda[:3, i][:, np.newaxis][[1, 0, 2]]
            ddmu[i-1, i] = np.array([[-1], [1], [1]]) * dda[:3, i][:, np.newaxis][[1, 0, 2]]
        return ddmu