import torch
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
from multidag.common.consts import DEVICE, OPTIMIZER, NONLINEARITIES
from multidag.common.utils import weights_init
from multidag.dag_utils import h_W, count_accuracy, is_dag
from multidag.model import LSEM
import os
import numpy as np


D_Loss = torch.nn.BCEWithLogitsLoss(reduction='mean')


def MSE(w1, w2):
    assert len(w1.shape) == 3
    assert len(w2.shape) == 3
    m = w1.shape[0]
    return ((w1 - w2) ** 2).view(m, -1).sum(dim=-1).mean()

def W_dist(w1, w2, norm='l1'):
    assert len(w1.shape) == 3
    assert len(w2.shape) == 3
    m = w1.shape[0]

    if norm == 'l1':
        d = torch.abs(w1 - w2)
    elif norm == 'l2':
        d = (w1 - w2) ** 2
    else:
        print('not supported')
        return 0

    return d.view(m, -1).sum(dim=-1).mean()


class MLP(nn.Module):
    def __init__(self, hidden_dims, nonlinearity='relu'):
        super(MLP, self).__init__()
        hidden_dims = list(map(int, hidden_dims.split("-")))
        # output size is 1
        hidden_dims.append(1)

        layers = []
        activation_fns = []
        # input size is 1
        prev_size = 1
        for h in hidden_dims:
            layers.append(nn.Linear(prev_size, h))
            prev_size = h
            activation_fns.append(NONLINEARITIES[nonlinearity])

        self.layers = nn.ModuleList(layers)
        self.activation_fns = nn.ModuleList(activation_fns)
        weights_init(self)

    def forward(self, x):
        for l, layer in enumerate(self.layers):
            x = layer(x)
            if l + 1 < len(self.layers):
                x = self.activation_fns[l](x)
        return x


class Trainer:
    def __init__(self, se, gn, g_dag, optimizer, data_base, constraint_type='notears',
                 K_mask=None, hyperparameters={}, hidden_dims='32-32', nonlinearity='relu'):
        self.se = se
        self.gn = gn
        self.g_dag = g_dag
        self.db = data_base
        if K_mask is None:
            self.K_mask = np.arange(self.db.K)
            self.K_mask = np.arange(self.db.K)
        else:
            self.K_mask = K_mask

        self.optimizer = optimizer
        self.train_itr = 0
        self.constraint_type = constraint_type

        # TODO hyperparameters may be different
        self.hyperparameter = dict()
        default = {'rho': 0.1, 'lambda': 1.0, 'c': 1.0, 'eta': 0.5, 'gamma': 1e-4,
                   'mu': 10.0, 'dual_interval': 50, 'init':  1}

        for key in default:
            if key in hyperparameters:
                self.hyperparameter[key] = hyperparameters[key]
            else:
                self.hyperparameter[key] = default[key]

        self.n = data_base.n

        # initialize lambda and alpha
        self.gamma = self.hyperparameter['gamma']
        self.rho = self.hyperparameter['rho']
        self.ld = torch.ones(size=[len(K_mask)]).to(DEVICE) * self.hyperparameter['lambda']
        self.c = torch.ones(size=[len(K_mask)]).to(DEVICE) * self.hyperparameter['c']

        # MLP layers
        # the input output dimensions are both 1.
        self.mlp_layers = MLP(hidden_dims=hidden_dims, nonlinearity=nonlinearity).to(DEVICE)
        self.opt_mlp = OPTIMIZER['adam'](self.mlp_layers.parameters(),
                                         lr=1e-3,
                                         weight_decay=1e-5)

    def train(self, epochs, start_epoch=0, loss_type=None):
        """
        training logic
        """
        progress_bar = tqdm(range(start_epoch, start_epoch + epochs))
        dsc = ''
        batch_size = min([self.n, 1000 // len(self.K_mask)])
        X = self.db.load_data(batch_size=batch_size, device=DEVICE)[self.K_mask]
        for e in progress_bar:
            self._train_epoch(e, X, progress_bar, dsc, loss_type)
        print('final use rho = {:.4f}'.format(self.rho))
        return self.save()

    def _train_epoch(self, epoch, X, progress_bar, dsc, loss_type, num_primal_steps=100):

        for _ in range(num_primal_steps):
            self.g_dag.train()
            self.mlp_layers.train()
            # -----------------
            #  primal step
            # -----------------
            self.optimizer.zero_grad()
            self.opt_mlp.zero_grad()

            loss, h_D, log = self.get_loss(X, self.ld, self.c)
            loss.backward()

            self.optimizer.step()
            self.opt_mlp.step()

            self.g_dag.proximal_update(self.gamma)
        # -----------------
        #  dual step
        # -----------------
        if (epoch + 1) % self.hyperparameter['dual_interval'] == 0:
            self.gamma *= 0.99
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.99
            if log['SE'] / self.se > log['l1/l2'] / self.gn + 0.05:
                self.rho *= 0.9
            elif log['SE'] / self.se < log['l1/l2'] / self.gn - 0.05:
                self.rho *= 1.1
            self.ld = torch.clamp(self.ld + self.c * (1e3 - (1e3 - h_D) * ((1e3 - h_D) > 0)), min=0, max=1e12)
            self.c = torch.clamp(self.c * (1 + self.hyperparameter['eta']), min=0, max=1e15)

        # info
        progress_bar.set_description("[SE: %.2f] [l1/l2: %.2f] [hw: %.2f] [conn: %.2f, one: %.2f] [ld: %.2f, c: %.2f]" %
                                     (log['SE'], log['l1/l2'], log['h_D']*1e8, log['conn']*1e8, log['one'],
                                      self.ld.mean().item(), self.c.mean().item()) + dsc)
        return

    def get_loss(self, X, ld, c):
        G_D = self.g_dag.G * self.g_dag.T

        # Version 1: G in last layer
        X = X.unsqueeze(dim=-1)  # [K, n, d, 1] tensor
        # apply a few non-linear layers
        f_X = self.mlp_layers(X).squeeze(dim=-1)  # [K, n, d] tensor
        G_f_X = torch.einsum('bji,bnj->bnij', G_D, f_X)  # [K, n, d, d] tensor
        f_GX = torch.sum(G_f_X, dim=-1)  # [K, n, d] tensor
        X = X.squeeze()

        # # Version 2: G in first layer
        # # Get G_D * X
        # GX = torch.einsum('bji,bnj->bnij', G_D, X)  # [K, n, d, d] tensor
        # GX = torch.sum(GX, dim=-1, keepdim=True)  # [K, n, d, 1]
        # # Add a few non-linear layers
        # f_GX = self.mlp_layers(GX.float()).squeeze(dim=-1)  # [K, n, d]

        # Squared Error
        loss_se = ((X - f_GX) ** 2).sum([2]).mean()

        # dagness constraints
        one = (self.g_dag.T - 1).abs().mean()
        mu_one = self.hyperparameter['mu'] * one

        conn = h_W[self.constraint_type](self.g_dag.T)
        lambda_conn = ld.mean() * conn * self.db.p
        c_conn_2 = 0.5 * c.mean() * conn ** 2 * self.db.p

        h_D = h_W[self.constraint_type](G_D)
        # if h_D.sum().item() == 0:
        #     for i in range(len(h_D)):
        #         assert is_dag(G_D[i].detach().numpy())
        lambda_h_wD = (ld * h_D).mean()   # lagrangian term
        c_hw_2 = 0.5 * (c * h_D * h_D).mean()  # l2 penalty

        # group norm
        w_l1_l2 = torch.linalg.norm(G_D, ord=2, dim=0).sum()
        rho_w_l1 = self.rho * np.sqrt(self.db.p * np.log(self.db.p) / self.db.n /
                   X.shape[0] ** 2) * w_l1_l2

        loss = loss_se + lambda_conn + c_conn_2 + mu_one + rho_w_l1 + lambda_h_wD + c_hw_2

        log = {'SE': loss_se.item(),
               'l1/l2': w_l1_l2.item(),
               'h_D': h_D.mean().item(),
               'conn': conn.item(),
               'one': one.item()
               }
        return loss, h_D.mean().item(), log

    def save(self):
        return {'G': self.g_dag.G.detach().cpu().numpy(), 'T': self.g_dag.T.detach().cpu().numpy()}

    # def evaluate(self):
    #     G_true = np.abs(np.sign(self.db.G[self.K_mask]))
    #     G_est = np.abs((self.g_dag.G * self.g_dag.T).detach().cpu().numpy())
    #     G_est[G_est < self.hyperparameter['threshold']] = 0
    #     G_est = np.sign(G_est)
    #     accs = []
    #     for k in range(G_true.shape[0]):
    #         acc = count_accuracy(G_true[k], G_est[k])
    #         accs.append(acc)
    #     return accs
