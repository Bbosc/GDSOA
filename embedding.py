import time
import torch
import numpy as np
from forward_kinematics import ForwardKinematic

class Embedding:
    def __init__(self, dimension: int, x: torch.Tensor, fk: ForwardKinematic):
        self.dim = dimension
        self.fk = fk
        self.x = x
        assert self.x.ndim == 3, "X must be 3 dimensional : #obstacles, cartesian_dim, 1"
        self._value = 0
        self.gradient = np.zeros((1, self.dim))
        self.hessian = np.zeros((self.dim, self.dim))
    
    def update_parameters(self, mu, sigma):
        self.nmu = mu[:, :, np.newaxis]
        self.diff = self.x[:, np.newaxis] - self.nmu[np.newaxis, :]
        self.nsigma = sigma
        self.gradient = np.zeros((1, self.dim))
        self.hessian = np.zeros((self.dim, self.dim))

    def profiler(func):
        def wrapper(*args):
            start = time.time()
            res = func(*args)
            print(f'execution frequency of {func.__name__}: {1/(time.time() - start):.4f} Hz')
            return res
        return wrapper

    def compute_value(self):
        prefix = 1/(np.sqrt(np.power(2*np.pi, self.dim) * np.linalg.det(self.nsigma)))
        exp = -0.5*np.einsum('bdij,djk,bdkn->bd', self.diff.transpose(0, 1, 3, 2), np.linalg.inv(self.nsigma), self.diff)
        res = prefix * np.exp(exp)
        return res
    
    def derive(self, q, dq, sigma):
        # update the value of the covariances and centroids
        mus, sigmas, dmus, dsigmas, ddmus, ddsigmas = self.fk(q, dq)
        self.update_parameters(mu=mus, sigma=sigmas)
        p = sigma * self.compute_value()
        sigma_inv = np.linalg.inv(self.nsigma)
        dsigma_inv = np.einsum('kmn, knpo, kpq -> kmqo', -sigma_inv, dsigmas, sigma_inv)
        dpdmu = self._derive_wrt_mu_m(p)
        dpdsigma = self._derive_wrt_sigma_m(p)
        self.gradient = (dpdmu @ dmus).squeeze(2) + np.einsum('nkij, kijd -> nkd', dpdsigma, dsigmas)
        hessian = np.einsum('nkijd, kjig -> nkdg', self._derive_wrt_q_mu_m(p, sigma_inv, dsigma_inv, dmus), dmus[:, :, np.newaxis]) + \
            np.einsum('nkij, kdgj->nkdg', dpdmu, ddmus) + \
            np.einsum('nkijd, kijp->nkdp', self._derive_wrt_q_sigma_m(p, sigma_inv, dsigma_inv, dmus), dsigmas) + \
            np.einsum('nkij,kdgij->nkdg', dpdsigma, ddsigmas)
        self.hessian = hessian.sum(0).sum(0)
        return self.gradient.sum(1).sum(0, keepdims=True), self.hessian
    
    def value_only(self, q):
        self.fk(q=q, dq=np.zeros_like(q), derivation_order=0)
        self.update_parameters(mu=self.fk.mus, sigma=self.fk.sigmas)
        p = self.compute_value()
        return p.sum()
    
    def _derive_wrt_mu(self, p, mu, sigma):
        return p * np.linalg.inv(sigma) @ (self.x - mu)

    def _derive_wrt_sigma(self, p, mu, sigma):
        inv = np.linalg.inv(sigma)
        d = self.x - mu
        return 0.5 * p * (inv @ d @ d.transpose((0, 2, 1)) @ inv)
    
    def _derive_wrt_mu_m(self, p):
        return np.einsum('nk, nkij->nkji', p, np.linalg.inv(self.nsigma) @ self.diff)

    def _derive_wrt_sigma_m(self, p):
        inv = np.linalg.inv(self.nsigma)
        return 0.5 * np.einsum('nk, nkij->nkij', p, (inv @ self.diff @ self.diff.transpose(0, 1, 3, 2) @ inv))
    
    
    def _derive_wrt_q_sigma_m(self, p, sigma_inv, dsigma_inv, dmus):
        d = self.diff.copy()
        a = 0.5 * np.einsum('nkd, nkij->nkijd', self.gradient, (sigma_inv @ d @ d.transpose((0, 1, 3, 2)) @ sigma_inv))
        b1 = np.einsum('kijd, nkjq->nkiqd', dsigma_inv, (self.diff @ self.diff.transpose(0, 1, 3, 2) @ sigma_inv))
        b2 = np.einsum('nkij, kjqd->nkiqd', sigma_inv @ self.diff @ self.diff.transpose(0, 1, 3, 2), dsigma_inv)
        b3 = np.einsum('kid, nkpq->nkiqd', -sigma_inv@dmus, self.diff.transpose(0, 1, 3, 2) @ sigma_inv)
        b4 = np.einsum('nkij, kdp->nkipd', -sigma_inv@self.diff, dmus.transpose(0, 2, 1) @ sigma_inv)
        b = 0.5 * np.einsum('nk, nkijd -> nkijd', p, b1 + b2 + b3 + b4)
        return a + b

    def _derive_wrt_q_mu(self, gradient, p, sigma_inv, dsigma_inv, mu, dmu):
        a = np.einsum('bq,bij->qi', gradient, sigma_inv @ (self.x - mu))
        b = p * dsigma_inv.transpose(2, 1, 0) @ (self.x - mu).squeeze(0)
        c = - p * sigma_inv @ dmu
        return a + b.squeeze() + c.T

    def _derive_wrt_q_mu_m(self, p, sigma_inv, dsigma_inv, dmus):
        a = np.einsum('nkd, nkij->nkjid', self.gradient, np.linalg.inv(self.nsigma) @ self.diff)
        b = np.einsum('nk, kijd, nkjf -> nkfid', p, dsigma_inv, self.diff)
        c = - np.einsum('nk, kid -> nkid', p, sigma_inv @ dmus)[:, :, np.newaxis]
        return a + b + c   

    def _derive_wrt_q_sigma(self, gradient, p, sigma_inv, dsigma_inv, mu, dmu):
        d = self.x - mu
        a = 0.5 * np.einsum('bq,bij->ijq', gradient, (sigma_inv @ d @ d.transpose((0, 2, 1)) @ sigma_inv))
        b1 = np.einsum('mnk, bnp -> mpk', dsigma_inv, d @ d.transpose(0, 2, 1) @ sigma_inv)
        b2 = np.einsum('bmn,npk->mpk', sigma_inv @ d @ d.transpose(0, 2, 1), dsigma_inv)
        b3 = np.einsum('mk, bpn->mnk', -sigma_inv @ dmu, d.transpose(0,2, 1) @ sigma_inv)
        b4 = np.einsum('bmp, kn->mnk', -sigma_inv @ d, dmu.T @ sigma_inv)
        b = 0.5 * p * (b1 + b2 + b3 + b4)
        return a + b



    @property
    def value(self)->torch.Tensor:
        return self.compute_value()
