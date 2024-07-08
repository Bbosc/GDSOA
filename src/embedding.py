import numpy as np
from .forward_kinematics import ForwardKinematic



class Embedding:
    def __init__(self, embeddings: list)->None:
        self._embeddings = embeddings
        self._value = None

    def __call__(self, **kwargs) -> tuple:
        psi = 0
        q = kwargs["q"]
        dq = kwargs["dq"]
        gradient = np.zeros((1, q.shape[0]))
        hessian = np.zeros((q.shape[0], q.shape[0]))
        for embedding in self._embeddings:
            psi+= embedding.value(**kwargs)
            gradient += embedding.gradient()
            hessian += embedding.hessian()
        return psi, gradient, hessian

    def value(self, **kwargs)->np.ndarray:
        psi=0
        for embedding in self._embeddings:
            psi += embedding.value(**kwargs)
        return psi


class Collision:
    def __init__(self, x: np.ndarray, fk: ForwardKinematic):
        self.x = x
        assert self.x.ndim == 3, "X must be 3 dimensional : #obstacles, cartesian_dim, 1"
        self.fk = fk

    def _update_parameters(self, q: np.ndarray, dq: np.ndarray, derivation_order: int = 2):
        mus, sigmas, dmus, dsigmas, ddmus, ddsigmas = self.fk(q, dq, derivation_order=derivation_order)
        self.sigma = sigmas
        self.mu = mus[:, :, np.newaxis]
        self.diff = self.x[:, np.newaxis] - self.mu[np.newaxis, :]
        self.dmus = dmus
        self.dsigmas = dsigmas
        self.ddmus = ddmus
        self.ddsigmas = ddsigmas

    def value(self, **kwargs)->np.ndarray:
        q = kwargs['q']
        dq = kwargs['dq']
        self._update_parameters(q, dq, derivation_order=kwargs["derivation_order"])
        self._p = self._collision_probability()
        return self._p

    def gradient(self)->np.ndarray:
        return self._first_derivative(self._p).sum(1).sum(0, keepdims=True)

    def hessian(self)->np.ndarray:
        return self._second_derivative(self._p).sum(0).sum(0)

    def _collision_probability(self)->np.ndarray:
        prefix = 1/(np.sqrt(np.power(2*np.pi, self.x.shape[1]) * np.linalg.det(self.sigma)))
        exp = -0.5*np.einsum('bdij,djk,bdkn->bd', self.diff.transpose(0, 1, 3, 2), np.linalg.inv(self.sigma), self.diff)
        res = prefix * np.exp(exp)
        return res / self.x.shape[0] * self.fk.priors

    def _first_derivative(self, p: np.ndarray)->np.ndarray:
        self.__dpdmu = self._derive_wrt_mu(p)
        self.__dpdsigma = self._derive_wrt_sigma(p)
        self._gradient = (self.__dpdmu @ self.dmus).squeeze(2) + np.einsum('nkij, kijd -> nkd', self.__dpdsigma, self.dsigmas)
        return self._gradient

    def _second_derivative(self, p: np.ndarray)->np.ndarray:
        sigma_inv = np.linalg.inv(self.sigma)
        dsigma_inv = np.einsum('kmn, knpo, kpq -> kmqo', -sigma_inv, self.dsigmas, sigma_inv)
        hessian = np.einsum('nkijd, kjig -> nkdg', self._derive_wrt_q_mu(p, sigma_inv, dsigma_inv, self.dmus), self.dmus[:, :, np.newaxis]) + \
            np.einsum('nkij, kdgj->nkdg', self.__dpdmu, self.ddmus) + \
            np.einsum('nkijd, kijp->nkdp', self._derive_wrt_q_sigma(p, sigma_inv, dsigma_inv, self.dmus), self.dsigmas) + \
            np.einsum('nkij,kdgij->nkdg', self.__dpdsigma, self.ddsigmas)
        return hessian
    
    def _derive_wrt_mu(self, p):
        return np.einsum('nk, nkij->nkji', p, np.linalg.inv(self.sigma) @ self.diff)

    def _derive_wrt_sigma(self, p):
        inv = np.linalg.inv(self.sigma)
        return 0.5 * np.einsum('nk, nkij->nkij', p, (inv @ self.diff @ self.diff.transpose(0, 1, 3, 2) @ inv))
    
    def _derive_wrt_q_sigma(self, p, sigma_inv, dsigma_inv, dmus):
        d = self.diff.copy()
        a = 0.5 * np.einsum('nkd, nkij->nkijd', self._gradient, (sigma_inv @ d @ d.transpose((0, 1, 3, 2)) @ sigma_inv))
        b1 = np.einsum('kijd, nkjq->nkiqd', dsigma_inv, (self.diff @ self.diff.transpose(0, 1, 3, 2) @ sigma_inv))
        b2 = np.einsum('nkij, kjqd->nkiqd', sigma_inv @ self.diff @ self.diff.transpose(0, 1, 3, 2), dsigma_inv)
        b3 = np.einsum('kid, nkpq->nkiqd', -sigma_inv@dmus, self.diff.transpose(0, 1, 3, 2) @ sigma_inv)
        b4 = np.einsum('nkij, kdp->nkipd', -sigma_inv@self.diff, dmus.transpose(0, 2, 1) @ sigma_inv)
        b = 0.5 * np.einsum('nk, nkijd -> nkijd', p, b1 + b2 + b3 + b4)
        return a + b

    def _derive_wrt_q_mu(self, p, sigma_inv, dsigma_inv, dmus):
        a = np.einsum('nkd, nkij->nkjid', self._gradient, np.linalg.inv(self.sigma) @ self.diff)
        b = np.einsum('nk, kijd, nkjf -> nkfid', p, dsigma_inv, self.diff)
        c = - np.einsum('nk, kid -> nkid', p, sigma_inv @ dmus)[:, :, np.newaxis]
        return a + b + c   

    def distance_metric(self):
        distances = np.linalg.norm(self.x[np.newaxis, :] - self.mu[:, np.newaxis, :], axis=2)
        return np.array([np.min(distance) for distance in np.split(distances, self.fk.robot_model.partitions[:-1])])


class JointLimit:
    def __init__(self, limits: np.ndarray, slope: float = 30, booster: int = 1000)->None:
        self._limits = limits
        self._slope = slope
        self._booster = booster
        self._value = None
        self._gradient = None
        self._hessian = None

    def value(self, **kwargs)->np.ndarray:
        self._q = kwargs['q'] 
        upper_bound = 1 / (1 + np.exp(self._slope*(self.limits[1] - self._q)))
        lower_bound = 1 / (1 + np.exp(self._slope*(self._q - self.limits[0])))
        return self._booster*(upper_bound + lower_bound)

    def gradient(self)->np.ndarray:
        return self._first_derivative(self._q)

    def hessian(self)->np.ndarray:
        return np.diag(self._second_derivative(self._q))

    def _first_derivative(self, q: np.ndarray):
        u = (1 + np.exp(self._slope*(self.limits[1] - q)))
        u_p = - self._slope * np.exp(self._slope*(self.limits[1] - q))
        one = - u_p / (u**2)
        u = 1 + np.exp(self._slope*(q - self.limits[0]))
        u_p = self._slope * np.exp(self._slope*(q - self.limits[0]))
        two = - u_p / (u**2)
        return self._booster*(one + two)

    def _second_derivative(self, q: np.ndarray):
        u = self._slope * np.exp(self._slope*(self.limits[1] - q))
        v = (1 + np.exp(self._slope*(self.limits[1] - q)))**2
        u_p = -(self._slope**2)*np.exp(self._slope*(self.limits[1] - q))
        v_p = - 2 * (1 + np.exp(self._slope*(self.limits[1] - q))) * self._slope * np.exp(self._slope*(self.limits[1] - q))
        one = (u_p * v - u * v_p)/(v**2)

        u = self._slope * np.exp(self._slope*(q - self.limits[0]))
        v = (1 + np.exp(self._slope*(q - self.limits[0])))**2
        u_p = (self._slope**2)*np.exp(self._slope*(q - self.limits[0]))
        v_p = 2 * (1 + np.exp(self._slope*(q - self.limits[0]))) * self._slope * np.exp(self._slope*(q - self.limits[0]))
        two = - (u_p * v - u * v_p)/(v**2)
        return self._booster*(one + two)
