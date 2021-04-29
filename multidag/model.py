import torch.nn as nn
import torch
from multidag.common.consts import DEVICE
from multidag.common.utils import weights_init
from torch.nn.parameter import Parameter


class G_DAG(nn.Module):
    def __init__(self, num_dags, p):
        super(G_DAG, self).__init__()
        self.p = p
        self.K = num_dags
        self._g = Parameter(torch.rand(size=[num_dags, p, p]))
        weights_init(self)

    @property 
    def w(self):
        # make diagonal zero
        return self._w * (1 - torch.eye(self.d).unsqueeze(0).repeat(self.m, 1, 1).to(DEVICE))
        
    def forward(self, idx):
        return self.w[idx]


class LSEM:
    @staticmethod
    def forward(W, X):
        """
        :param W: [batch, d, d] tensor  d1 == d
        :param X: [batch, n, d] tensor
        """
        # batch, n, d = X.shape
        WX = torch.einsum('bji,bnj->bnij', W, X)  # [batch, n, d, d] tensor
        mean = torch.sum(WX, dim=-1)  # [batch, n, d]

        return mean

    @staticmethod
    def SE(W, X):
        """
        sequared error
        """
        mean = LSEM.forward(W, X)  # [batch, n, d]
        se = (X - mean) ** 2  # [batch, n, d]
        return se
