from typing import Literal
import numpy as np
from .embedding import Embedding


class DynamicalSystem:
    def __init__(self, stiffness: np.ndarray, dissipation: np.ndarray, attractor: np.ndarray, embedding: Embedding, dt: float = 0.02) -> None:
        self.stiffness = stiffness
        self.dissipation = dissipation
        self.attractor = attractor
        self.embedding = embedding
        self.dt = dt
        self.x_logger = []

    def __call__(self, x, dx, mode:Literal['normal', 'geodesic', 'harmonic']='normal', **kwargs):
        if mode == 'normal':
            ddx = self.compute_basic_switched_acceleration(x.copy(), dx.copy(), **kwargs)
        elif mode == 'geodesic':
            ddx = self.compute_geodesic_acceleration(x.copy(), dx.copy())
        elif mode == 'harmonic':
            ddx = self.compute_harmonic_acceleration(x.copy(), dx.copy())
        else: raise NotImplementedError
        return self.integrate(x.copy(), dx.copy(), ddx)

    def integrate(self, x, dx, ddx):
        new_dx = dx + ddx * self.dt
        new_x = x + new_dx * self.dt
        # loggers
        self.x_logger.append(new_x)
        return new_x, new_dx
    
    def compute_harmonic_acceleration(self, x, dx):
        embedding, embedding_gradient, embedding_hessian = self.embedding.derive(x, dx)
        metric = self.compute_metric(embedding_gradient)
        harmonic = - np.linalg.inv(metric) @ self.stiffness @ (x - self.attractor) - np.linalg.inv(metric) @ self.dissipation @ dx
        return harmonic

    def compute_geodesic_acceleration(self, x, dx):
        embedding, embedding_gradient, embedding_hessian = self.embedding.derive(x, dx)
        metric = self.compute_metric(embedding_gradient)
        christoffel = self.compute_christoffel(metric, embedding_gradient, embedding_hessian)
        geodesic = - np.einsum('qij,i->qj', christoffel, dx) @ dx
        return geodesic

    def compute_basic_switched_acceleration(self, x, dx, **kwargs):
        def switch(psi, kappa: float):
            return 1 if psi <= kappa else 0
        embedding, embedding_gradient, embedding_hessian = self.embedding.derive(x, dx)
        metric = self.compute_metric(embedding_gradient)
        christoffel = self.compute_christoffel(metric, embedding_gradient, embedding_hessian)
        harmonic = - np.linalg.inv(metric) @ (self.stiffness @ (x - self.attractor) + metric @ self.dissipation @ dx)
        geodesic = - np.einsum('qij,i->qj', christoffel, dx) @ dx
        sigma = switch(psi=embedding.sum(), kappa=kwargs['kappa'])
        return sigma * harmonic + (1-sigma)*geodesic

    def compute_metric(self, embedding_gradient)->np.ndarray:
        d = embedding_gradient.shape[1]
        return np.eye(d) + np.outer(embedding_gradient, embedding_gradient)

    def compute_christoffel(self, metric, embedding_gradient: np.ndarray, embedding_hessian: np.ndarray):
        dm = self.derive_metric(embedding_gradient, embedding_hessian).transpose(0, 2, 1)
        im = np.linalg.inv(metric)
        return 0.5 * (np.einsum('qm,mji->qji', im, dm + dm.transpose(0,2,1)) - np.einsum('qm,ijm->qij', im, dm))
        
    @staticmethod
    def derive_metric(embedding_gradient: np.linalg.inv, embedding_hessian: np.linalg.inv)->np.ndarray:
        return np.einsum('pq,r->pqr', embedding_hessian, embedding_gradient.squeeze()) + np.einsum('p,qr->pqr', embedding_gradient.squeeze(), embedding_hessian)
    
    @staticmethod
    def generalized_sigmoid(x, b=1., a=0., k=1., m=0.):
        return (k-a) / (1 + np.exp(-b*(x-m))) + a
