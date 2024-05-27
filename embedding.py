import time
import torch
import numpy as np
from forward_kinematics import ForwardKinematic

class Embedding:
    def __init__(self, dimension: int, x: torch.Tensor, fk: ForwardKinematic):
        self.dim = dimension
        self.fk = fk
        self.x = x
        self._value = 0
        self.gradient = np.zeros((1, self.dim))
        self.hessian = np.zeros((self.dim, self.dim))
    
    def update_parameters(self, mu, sigma):
        self.nmu = mu[:, :, np.newaxis]
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
        diff = (self.x - self.nmu).reshape((self.nmu.shape[0], self.nmu.shape[1], 1))
        exp = -0.5 * diff.transpose(0, 2, 1) @ np.linalg.inv(self.nsigma) @ diff
        res = prefix * np.exp(exp).ravel()
        return res
    
    def derive(self, q, dq):
        # update the value of the covariances and centroids
        mus, sigmas, dmus, dsigmas, ddmus, ddsigmas = self.fk(q, dq)
        self.update_parameters(mu=mus, sigma=sigmas)
        p = self.compute_value()
        sigma_inv = np.linalg.inv(self.nsigma)
        dsigma_inv = np.einsum('kmn, knpo, kpq -> kmqo', -sigma_inv, dsigmas, sigma_inv)
        for i in range(p.shape[0]):
        # compute the gradient of the embedding
            dpdmu = self._derive_wrt_mu(p[i], self.nmu[i], self.nsigma[i])
            dpdsigma = self._derive_wrt_sigma(p[i], self.nmu[i], self.nsigma[i])
            gradient = (dpdmu.transpose(0, 2, 1) @ dmus[i]).squeeze() + np.einsum('bij,ijk->bk', dpdsigma, dsigmas[i])
        # compute the hessian of the embedding
            a = self._derive_wrt_q_mu(gradient, p[i], sigma_inv[i], dsigma_inv[i], self.nmu[i], dmus[i])
            b = self._derive_wrt_q_sigma(gradient, p[i], sigma_inv[i], dsigma_inv[i], self.nmu[i], dmus[i])
            hessian = a @ dmus[i] + np.einsum('bi,ijk->jk', dpdmu.squeeze(-1), ddmus[i].transpose(2, 0, 1)) + np.einsum('ijk, kjl->il', b.transpose(2, 0, 1), dsigmas[i]) + np.einsum('ijk,pqjk->pq', dpdsigma, ddsigmas[i])
            self.gradient  += gradient
            self.hessian  += hessian
        return self.gradient, self.hessian
    
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
    
    def _derive_wrt_q_mu(self, gradient, p, sigma_inv, dsigma_inv, mu, dmu):
        a = np.einsum('bq,bij->qi', gradient, sigma_inv @ (self.x - mu))
        b = p * dsigma_inv.transpose(2, 1, 0) @ (self.x - mu).squeeze(0)
        c = - p * sigma_inv @ dmu
        return a + b.squeeze() + c.T
    
    def _derive_wrt_q_sigma(self, gradient, p, sigma_inv, dsigma_inv, mu, dmu):
        d = self.x - mu
        a = 0.5 * np.einsum('bq,bij->ijq', gradient, (-sigma_inv + sigma_inv @ d @ d.transpose((0, 2, 1)) @ sigma_inv))
        b1 = np.einsum('mnk, bnp -> mpk', dsigma_inv, d @ d.transpose(0, 2, 1) @ sigma_inv)
        b2 = np.einsum('bmn,npk->mpk', sigma_inv @ d @ d.transpose(0, 2, 1), dsigma_inv)
        b3 = np.einsum('mk, bpn->mnk', -sigma_inv @ dmu, d.transpose(0,2, 1) @ sigma_inv)
        b4 = np.einsum('bmp, kn->mnk', -sigma_inv @ d, dmu.T @ sigma_inv)
        b = 0.5 * p * (b1 + b2 + b3 + b4)
        return a + b

    @property
    def value(self)->torch.Tensor:
        return self.compute_value()
