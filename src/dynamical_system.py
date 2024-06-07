import time
import numpy as np
from .embedding import Embedding


class DynamicalSystem:
    def __init__(self, stiffness: np.ndarray, dissipation: np.ndarray, attractor: np.ndarray, embedding: Embedding, dt: float = 0.02) -> None:
        self.stiffness = stiffness
        self.dissipation = dissipation
        self.attractor = attractor
        self.embedding = embedding
        self.speed_limits = np.ones_like(attractor) * 2.62
        self.dt = dt
        self.embedding_logger = []
        self.gradient_logger = []
        self.hessian_logger = []
        self.christoffel_logger = []
        self.metric_logger = []
        self.forces_logger = []
        self.x_logger = []
        self.dx_logger = []
        self.ddx_logger = []
        self.correction_logger = []
        self.keep_projection = False

    def __call__(self, x, dx):
        ddx = self.compute_acceleration(x.copy(), dx.copy())
        return self.integrate(x.copy(), dx.copy(), ddx)

    def profiler(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            res = func(*args, **kwargs)
            print(f'execution frequency of {func.__name__}: {1/(time.time() - start):.4f} Hz')
            return res
        return wrapper

    def compute_acceleration(self, x, dx):
        embedding, embedding_gradient, embedding_hessian = self.embedding.derive(x, dx)
        metric = self.compute_metric(embedding_gradient)
        christoffel = self.compute_christoffel(metric, embedding_gradient, embedding_hessian)
        harmonic = - np.linalg.inv(metric) @ self.stiffness @ (x - self.attractor) - self.dissipation @ dx
        geodesic = - np.einsum('qij,i->qj', christoffel, dx) @ dx

        # loggers
        self.embedding_logger.append(embedding)
        self.gradient_logger.append(embedding_gradient)
        self.hessian_logger.append(embedding_hessian)
        self.christoffel_logger.append(christoffel)
        self.metric_logger.append(metric)
        self.forces_logger.append(self.derive_metric(embedding_gradient, embedding_hessian).transpose(0, 2, 1))

        return geodesic
    
    def integrate(self, x, dx, ddx):
        metric = self.metric_logger[-1]
        _, vectors = np.linalg.eigh(metric)
        p = self.embedding_logger[-1].sum()
        # projection_weight = self.generalized_sigmoid(x=p, b=20, a=0, k=1, m=1.)
        # projected_speed = vectors[0] * np.dot(new_dx, vectors[0])/np.linalg.norm(vectors[0])
        projected_accel = vectors[1] * np.dot(ddx, vectors[1])/np.linalg.norm(vectors[1])
        # new_dx = projection_weight * projected_speed + (1-projection_weight) * new_dx
        if p > 0.4:
            ddx = projected_accel
        new_dx = dx + ddx * self.dt
        new_x = x + new_dx * self.dt

        # loggers
        self.ddx_logger.append(ddx)
        self.dx_logger.append(new_dx)
        self.x_logger.append(new_x)
        return new_x, new_dx

    def compute_metric(self, embedding_gradient)->np.ndarray:
        d = embedding_gradient.shape[1]
        return np.eye(d) + np.outer(embedding_gradient, embedding_gradient)

    def compute_christoffel(self, metric, embedding_gradient: np.ndarray, embedding_hessian: np.ndarray):
        dm = self.derive_metric(embedding_gradient, embedding_hessian).transpose(0, 2, 1)
        im = np.linalg.inv(metric)
        return 0.5 * (np.einsum('qm,mji->qji', im, dm + dm.transpose(0,2,1)) - np.einsum('qm,ijm->qij', im, dm))
        
    def new_christoffel(self, metric, embedding_gradient, embedding_hessian, dx):
        dm = self.new_derive_metric(embedding_gradient.T, embedding_hessian).transpose(0, 2, 1)
        inv = np.linalg.inv(metric)
        T = np.zeros((2, 2, 2))
        for j in range(dx.shape[0]):
            for l in range(metric.shape[0]):
                for k in range(metric.shape[0]):
                    Tlk = 0
                    for m in range(dx.shape[0]):
                        Tlk += 0.5 * inv[j, m] * (dm[m, l, k] + dm[m, k, l] - dm[l, k, m])
                    T[j, l, k] = Tlk
        return T

    def new_geodesic(self, T, dx):
        ddx = np.zeros_like(dx)
        for j in range(dx.shape[0]):
            for l in range(dx.shape[0]):
                for k in range(dx.shape[0]):
                    ddx[j] += - T[j, l, k] * dx[l] * dx[k]
        return ddx

    def new_derive_metric(self, embedding_gradient, embedding_hessian):
        d = embedding_gradient.shape[0]
        return np.kron(embedding_hessian, embedding_gradient.T).reshape((d, d, d)) + np.kron(embedding_gradient, embedding_hessian.T).reshape((d, d, d))
                
    @staticmethod
    def derive_metric(embedding_gradient: np.linalg.inv, embedding_hessian: np.linalg.inv)->np.ndarray:
        return np.einsum('pq,r->pqr', embedding_hessian, embedding_gradient.squeeze()) + np.einsum('p,qr->pqr', embedding_gradient.squeeze(), embedding_hessian)
    
    def compute_dynamical_weights(self, x: np.ndarray, horizon: float = 0.005, discretion: int = 2):
        future_x = x + np.linspace(0, horizon, discretion)[:, np.newaxis].repeat(self.attractor.shape[0], axis=1) * (self.attractor - x)
        future_p = np.zeros((future_x.shape[0]))
        for i, q in enumerate(future_x):
            future_p[i] = self.embedding.value_only(q).sum()
        # weights = 1 if (np.sign(np.diff(future_p)[0]) != np.sign(np.diff(future_p)[1])) else 0
        # weights = 1 if (np.sign(np.diff(future_p)[0]) != np.sign(np.diff(future_p)[1])) else 0
        weights = 1 if np.diff(future_p) > 0.005 else 0
        # weights = self.generalized_sigmoid(np.diff(future_p), b=100, a=0, k=1., m=0.01)
        return weights, np.diff(future_p)

    @staticmethod
    def generalized_sigmoid(x, b=1., a=0., k=1., m=0.):
        c = -b*(x-m).item() # to avoid overflow in exp
        return (k-a) / (1 + np.exp(c)) + a
