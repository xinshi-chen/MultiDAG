import torch.nn as nn
import torch
import numpy as np
from multidag.common.consts import DEVICE
from multidag.common.utils import weights_init
from torch.nn.parameter import Parameter


class G_DAG(nn.Module):
    def __init__(self, num_dags, p):
        super(G_DAG, self).__init__()
        self.p = p
        self.K = num_dags
        self._G = Parameter(torch.randn(size=[num_dags, p, p], dtype=torch.double))
        self._T = Parameter(torch.rand(size=(1, p, p), dtype=torch.double))
        weights_init(self)

    @property
    def G(self):
        # make diagonal zero
        return self._G * (1 - torch.eye(self.p).unsqueeze(0).repeat(self.K, 1, 1).to(DEVICE))

    @property
    def T(self):
        return self._T.to(DEVICE)

    @T.setter
    def T(self, perm):
        B = np.tril(np.ones([self.p, self.p]), k=-1)
        target = perm.T.dot(B).dot(perm) + np.random.rand(self.p, self.p) * 0.05
        with torch.no_grad():
            self._T.data = torch.DoubleTensor(target).unsqueeze(0)
        print('### initialize T ###')

    def proximal_update(self, gamma):
        with torch.no_grad():
            G_norm = torch.linalg.norm(self.G, ord=2, dim=0, keepdim=True).detach() + 1e-20
            self._G.data = (self.G * torch.clamp(G_norm - gamma * self.T.abs(), min=0) / G_norm)
            self._T.data = (torch.sign(self.T) * torch.clamp(self.T.abs() - gamma * G_norm, min=0))

    def forward(self, idx):
        return self.g[idx]


class LSEM:
    @staticmethod
    def forward(G, X):
        """
        :param G: [K, d, d] tensor  d1 == d
        :param X: [K, n, d] tensor
        """
        # batch, n, d = X.shape
        GX = torch.einsum('bji,bnj->bnij', G, X)  # [batch, n, d, d] tensor
        mean = torch.sum(GX, dim=-1)  # [batch, n, d]

        return mean

    @staticmethod
    def SE(G, X):
        """
        se: squared error
        """
        mean = LSEM.forward(G, X)  # [K, n, d]
        se = ((X - mean) ** 2).sum([2]).mean()
        return se
