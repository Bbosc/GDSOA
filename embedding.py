import time
import torch
import numpy as np
from forward_kinematic import ForwardKinematic

class Embedding:
    def __init__(self, dimension: int, x: torch.Tensor, initial_mu: torch.Tensor, initial_sigma: torch.Tensor, fk: ForwardKinematic):
        self.dim = dimension
        self.fk = fk
        self.x = x
        self.nx = x.numpy()
        self._value = 0
        self._n_value = 0
        self.hessian_logger = []
        # self.gradient = torch.zeros(self.dim)
        # self.hessian = torch.zeros(self.dim, self.dim)
        self.update_parameters(initial_mu, initial_sigma)

    
    def update_parameters(self, mu, sigma):
        # self.mu = mu.repeat(self.x.shape[0], 1, 1)
        # self.sigma = sigma.repeat(self.x.shape[0], 1, 1)
        # self.nmu = mu.repeat((self.x.shape[0], 1, 1))
        # self.nsigma = sigma.repeat((self.x.shape[0], 1, 1))
        self.nmu = mu[:, :, np.newaxis]
        self.nsigma = sigma
        self.x = self.x
        self.nx = self.nx

    def profiler(func):
        def wrapper(*args):
            start = time.time()
            res = func(*args)
            print(f'execution frequency of {func.__name__}: {1/(time.time() - start):.4f} Hz')
            return res
        return wrapper

    # @profiler
    def compute_value(self):
        prefix = 1/(torch.sqrt(torch.tensor(2*torch.pi)).pow(self.dim) * torch.det(self.sigma))
        diff = self.x - self.mu
        exp = -0.5 * torch.bmm(diff.permute(0, 2, 1), torch.bmm(self.sigma.inverse(), diff))
        res = prefix * torch.exp(exp).ravel()
        return res.sum()

    # @profiler
    def compute_value_numpy(self):
        prefix = 1/(np.sqrt(np.power(2*np.pi, self.dim) * np.linalg.det(self.nsigma)))
        diff = (self.nx - self.nmu).reshape((self.nmu.shape[0], self.nmu.shape[1], 1))
        exp = -0.5 * diff.transpose(0, 2, 1) @ np.linalg.inv(self.nsigma) @ diff
        res = prefix * np.exp(exp).ravel()
        return res
    
    # @profiler
    def derive(self, q):
        # update the value of the covariances and centroids
        mus, sigmas, dmus, dsigmas, ddmus, ddsigmas = self.fk.test(q)
        self.update_parameters(mu=mus, sigma=sigmas)
        p = 10 * self.compute_value_numpy()
        gradients = np.zeros((1, p.shape[0]))
        hessians = np.zeros((gradients.shape[1], gradients.shape[1]))
        sigma_inv = np.linalg.inv(self.nsigma)
        for i in range(p.shape[0]):
        # compute the gradient of the embedding
            dpdmu = self._derive_wrt_mu(p[i], self.nmu[i], self.nsigma[i])
            dpdsigma = self._derive_wrt_sigma(p[i], self.nmu[i], self.nsigma[i])
            gradient = (dpdmu.transpose(0, 2, 1) @ dmus[i]).squeeze() + np.einsum('bij,ijk->bk', dpdsigma, dsigmas[i])
        # compute the hessian of the embedding
            dsigma_inv = np.zeros((sigmas[i].shape[0], sigmas[i].shape[1], gradient.shape[1]))
            for j in range(dsigma_inv.shape[-1]):
                dsigma_inv[:, :, j] = - sigma_inv[i] @ dsigmas[i, :, :, j] @ sigma_inv[i]
            # dsigma_inv = -np.einsum('ij,ijq,ij->qij', sigma_inv[i], dsigmas[i], sigma_inv[i])
            a = self._derive_wrt_q_mu(gradient, p[i], sigma_inv[i], dsigma_inv, self.nmu[i], dmus[i])
            b = self._derive_wrt_q_sigma(gradient, p[i], sigma_inv[i], dsigma_inv, self.nmu[i], dmus[i])
            hessian = a @ dmus[i] + np.einsum('bi,ijk->jk', dpdmu.squeeze(-1), ddmus[i].transpose(2, 0, 1)) + np.einsum('ijk, kjl->il', b.transpose(2, 0, 1), dsigmas[i]) + np.einsum('ijk,pqjk->pq', dpdsigma, ddsigmas[i])
            if i == 1:
                self.hessian_logger.append(hessian)
            gradients  += gradient
            hessians  += hessian
        return gradients, hessians
    
    def value_only(self, q):
        mus, sigmas, dmus, dsigmas, ddmus, ddsigmas = self.fk.test(q)
        self.update_parameters(mu=mus, sigma=sigmas)
        p = self.compute_value_numpy()
        return 10 *p.sum()


    # @profiler
    def _derive_wrt_mu(self, p, mu, sigma):
        return p * np.linalg.inv(sigma) @ (self.nx - mu)

    # @profiler 
    def _derive_wrt_sigma(self, p, mu, sigma):
        inv = np.linalg.inv(sigma)
        d = self.nx - mu
        return 0.5 * p * (-inv + inv @ d @ d.transpose((0, 2, 1)) @ inv)
    
    def _derive_wrt_q_mu(self, gradient, p, sigma_inv, dsigma_inv, mu, dmu):
        a = np.einsum('bq,bij->qi', gradient, sigma_inv @ (self.nx - mu))
        b = p * dsigma_inv.transpose(2, 1, 0) @ (self.nx - mu).squeeze(0)
        c = - p * sigma_inv @ dmu
        return a + b.squeeze() + c.T
    
    def _derive_wrt_q_sigma(self, gradient, p, sigma_inv, dsigma_inv, mu, dmu):
        d = self.nx - mu
        a = 0.5 * np.einsum('bq,bij->ijq', gradient, (-sigma_inv + sigma_inv @ d @ d.transpose((0, 2, 1)) @ sigma_inv))
        b1 = - dsigma_inv
        b2 = np.array([dsigma_inv[:, :, 0] @ d @ d.transpose((0, 2, 1)) @ sigma_inv, dsigma_inv[:, :, 1] @ d @ d.transpose((0, 2, 1)) @ sigma_inv]).squeeze().transpose(1, 2, 0)
        # b3 = - sigma_inv @ dmu @ d.transpose((0, 2, 1)) @ sigma_inv
        # b3 = np.einsum('ij,rq,brf,ij->qij', sigma_inv, dmu, d.transpose((0,2,1)), sigma_inv)
        # b3 = np.einsum('ij, n->inj', sigma_inv @ dmu, (d.squeeze(0).T @ sigma_inv).squeeze())
        b3 = np.array([sigma_inv @ d @ d.transpose(0, 2, 1) @ dsigma_inv[:, :, 0], sigma_inv @ d @ d.transpose(0, 2, 1) @ dsigma_inv[:, :, 1]]).squeeze().transpose(1, 2, 0)
        # b4 = - sigma_inv @ d @ dmu.T @ sigma_inv
        # b4 = np.einsum('ij,bfr,qr,ij->qij', sigma_inv, d, dmu.T, sigma_inv)
        # b4 = np.einsum('i, mn -> im', (sigma_inv @ d).squeeze(), (dmu.T @ sigma_inv).squeeze())
        b4 = np.array([-sigma_inv @ dmu[:, 0].reshape((d.shape)) @ d.transpose(0, 2, 1) @ sigma_inv, -sigma_inv @ dmu[:, 1].reshape(d.shape) @ d.transpose(0, 2, 1) @ sigma_inv]).squeeze().transpose(1,2,0)
        b5 = sigma_inv @ d @ d.transpose((0, 2, 1)) @ dsigma_inv.transpose(2, 1, 0)
        b5 = np.array([-sigma_inv @ d @ dmu[:, 0].reshape((d.shape)).transpose(0,2,1) @ sigma_inv, -sigma_inv @ d @ dmu[:, 1].reshape(d.shape).transpose(0, 2, 1) @ sigma_inv]).squeeze().transpose(1,2,0)
        b = 0.5 * p * (b1 + b2 + b3 + b4 + b5)
        return a + b

    @property
    def value(self)->torch.Tensor:
        return self.compute_value()
    
    @property
    def n_value(self):
        return self.compute_value_numpy()