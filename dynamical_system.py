import time
import numpy as np
from embedding import Embedding


class DynamicalSystem:
    def __init__(self, stiffness: np.ndarray, dissipation: np.ndarray, attractor: np.ndarray, embedding: Embedding, dt: float = 0.02) -> None:
        self.stiffness = stiffness
        self.dissipation = dissipation
        self.attractor = attractor
        self.embedding = embedding
        self.dt = dt
        self.gradient_logger = []
        self.weight_logger = []
        self.speed_logger = []
        self.christ_logger = []
        self.hessian_logger = []
        self.metric_logger = []
        self.gr_logger = []

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
        embedding_gradient, embedding_hessian = self.embedding.derive(x, dx, 1)
        sigma = self.compute_dynamical_weights(x, embedding_gradient, horizon=np.pi/1000)
        # embedding_gradient *= sigma
        # embedding_hessian *= sigma
        self.gradient_logger.append(embedding_gradient)
        self.hessian_logger.append(embedding_hessian)
        metric = self.compute_metric(embedding_gradient)
        self.metric_logger.append(metric)
        christoffel = self.compute_christoffel(metric, embedding_gradient, embedding_hessian)
        self.christ_logger.append(christoffel)
        # self.gr_logger.append(gr)
        harmonic = - np.linalg.inv(metric) @ self.stiffness @ (x - self.attractor) - self.dissipation @ dx
        geodesic = - np.einsum('qij,i->qj', christoffel, dx) @ dx
        self.speed_logger.append(dx)
        self.weight_logger.append(sigma)
        # return harmonic + geodesic
        return (1-sigma)*harmonic + sigma * geodesic
    
    def integrate(self, x, dx, ddx):
        new_dx = dx + ddx * self.dt
        new_x = x + new_dx * self.dt
        return new_x, new_dx

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
    
    def compute_dynamical_weights(self, x: np.ndarray, gradient, horizon: float = 0.005, discretion: int = 2):
        # future_x = x + np.linspace(0, horizon, discretion)[:, np.newaxis].repeat(self.attractor.shape[0], axis=1) * (self.attractor - x)
        # future_p = np.zeros((future_x.shape[0]))
        # for i, q in enumerate(future_x):
        #     future_p[i] = self.embedding.value_only(q).sum()
        # weights = 1 if (np.diff(future_p) > 0) else 0
        weights = self.generalized_sigmoid(np.linalg.norm(gradient, axis=1), b=10, a=0, k=1., m=10)
        return weights

    @staticmethod
    def generalized_sigmoid(x, b=1., a=0., k=1., m=0.):
        c = min(-b*(x-m).item(), 20) # to avoid overflow in exp
        return (k-a) / (1 + np.exp(c)) + a