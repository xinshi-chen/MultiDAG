import torch.nn as nn
import torch
import torch.nn.functional as F
from gan4dag.common.consts import DEVICE, NONLINEARITIES
from gan4dag.common.utils import weights_init
from gan4dag.dag_utils import project_to_dag, sampler
import networkx as nx
import numpy as np
from torch.nn.parameter import Parameter


class GenNet(nn.Module):

    def __init__(self, d):
        super(GenNet, self).__init__()
        self.d = d
        self.W = Parameter(torch.empty(size=[d, d]))
        self.noise_mean = Parameter(torch.empty(size=[d]))
        weights_init(self)
        self.V = Parameter(torch.ones(size=[d, d]))
        self.noise_sd = Parameter(torch.ones(size=[d]))

    def forward(self, Z):
        """
        :param Z: [d, d] or [m, d, d] tensor
        :return:
        """
        W = Z * self.V
        W = W + self.W

        return W

    def gen_one_set_X(self, n):

        with torch.no_grad():
            while 1:
                z = torch.normal(0, 1, size=(self.d, self.d)).to(DEVICE)
                w = self.forward(z).detach()
                _, P = project_to_dag(w.cpu().numpy(), sparsity=1.0, w_threshold=0.1)
                if P is not None:
                    P = torch.tensor(P).to(DEVICE).detach()
                    break
                else:
                    print('W is far from a dag and cannot project.')

        # The following part is differentiable
        W = self.forward(z.detach())
        W = W * P  # make it a DAG
        # TODO: should have one more layer to adjust the weight, in case the sparsity/threshold is not correct

        # samples
        X = sampler(W, n, self.noise_mean, self.noise_sd, noise_type='gauss')
        return X

# class DiscNet(nn.Module):
#
#     def __init__(self, d):


if __name__ == '__main__':
    d = 5
    m = 10
    gen = GenNet(d).to(DEVICE)
    Z = torch.normal(0, 1, size=[m, d, d]).to(DEVICE)
    W_gen = gen(Z)
    print(W_gen)


# class QNet(nn.Module):
#
#     def __init__(self, args, hidden_dims_eigs, activation='relu', bias_init='zero'):
#         super(QNet, self).__init__()
#
#         self.d = args.d
#         self.dim_in = args.dim_in
#         self.mu = args.mu
#         self.L = args.L
#         self.temp = args.temperature
#         if activation != 'none':
#             self.act_fcn = NONLINEARITIES[activation]
#
#         # generate diagonals
#         layers = []
#         hidden_dims_eigs = tuple(map(int, hidden_dims_eigs.split("-")))
#         prev_size = args.dim_in
#         for h in hidden_dims_eigs:
#             if h>0:
#                 layers.append(nn.Linear(prev_size, h, bias=True))
#                 prev_size = h
#         layers.append(nn.Linear(prev_size, self.d-2, bias=True))
#         self.layers_w = nn.ModuleList(layers)
#
#         weights_init(self, bias=bias_init)
#
#     def forward(self, u):
#         batch = u.shape[0]
#
#         # generate diagonals
#         for l, layer in enumerate(self.layers_w):
#             if l == 0:
#                 e = layer(u)
#             else:
#                 e = layer(e)
#             if l + 1 < len(self.layers_w):
#                 e = self.act_fcn(e)
#         # shift to [mu, L]
#         e = F.softmax(e, dim=-1)
#         e = e * (self.L - self.mu) + self.mu
#
#         # concat with mu and L
#         eigen1 = torch.ones([batch, 1]).to(DEVICE) * self.L
#         eigend = torch.ones([batch, 1]).to(DEVICE) * self.mu
#         eigens = torch.cat([eigen1.detach(), e, eigend.detach()], dim=-1)
#
#         return eigens
#
#
# def qeq(q, e):
#     batch, d = e.shape
#     qe = q * e.view(batch, 1, d)
#     w = torch.matmul(qe, q.transpose(-1, -2))
#     return w
#
#
# def qeq_inv(q, e):
#     batch, d = e.shape
#     e_inv = 1 / e
#     qe_inv = q * e_inv.view(batch, 1, d)
#     w_inv = torch.matmul(qe_inv, q.transpose(-1, -2))
#     return w_inv
