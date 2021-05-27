import torch
import torch.nn.functional as F
from tqdm import tqdm
from multidag.common.consts import DEVICE, OPTIMIZER
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


class Trainer:
    def __init__(self, se, gn, g_dag, optimizer, data_base, constraint_type='notears',
                 K_mask=None, hyperparameters={}):
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
                   'mu': 10.0, 'dual_interval': 50, 'init': 1, 'alpha': 0}

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

    def _train_epoch(self, epoch, X, progress_bar, dsc, loss_type):

        self.g_dag.train()

        # -----------------
        #  primal step
        # -----------------
        self.optimizer.zero_grad()

        loss, h_D, log = self.get_loss(X, self.ld, self.c)
        loss.backward()

        self.optimizer.step()
        self.g_dag.proximal_update(self.gamma)
        # -----------------
        #  dual step
        # -----------------
        if epoch > 2000 and (epoch + 1) % self.hyperparameter['dual_interval'] == 0:
            self.gamma *= 0.99
            for param_group in self.optimizer.param_groups:
                param_group['lr'] *= 0.99
            if log['SE'] / self.se > log['l1/l2'] / self.gn + 0.05:
                self.rho *= 0.98
            elif log['SE'] / self.se < log['l1/l2'] / self.gn - 0.05:
                self.rho *= 1.02
            self.ld = torch.clamp(self.ld + self.c * (1e3 - (1e3 - h_D) * ((1e3 - h_D) > 0)), min=0, max=1e12)
            self.c = torch.clamp(self.c * (1 + self.hyperparameter['eta']), min=0, max=1e15)

        # info
        progress_bar.set_description("[SE: %.2f] [l1/l2: %.2f] [hw: %.2f] [conn: %.2f, one: %.2f] [ld: %.2f, c: %.2f]" %
                                     (log['SE'], log['l1/l2'], log['h_D']*1e8, log['conn']*1e8, log['one'],
                                      self.ld.mean().item(), self.c.mean().item()) + dsc)
        return

    def get_loss(self, X, ld, c):
        G_D = self.g_dag.G * self.g_dag.T
        # Squared Error
        loss_se = LSEM.SE(G_D, X)

        # dagness constraints
        one = (self.g_dag.T - 1).abs().mean()
        mu_one = self.hyperparameter['mu'] * one

        conn = h_W[self.constraint_type](self.g_dag.T)
        lambda_conn = ld.mean() * conn
        c_conn_2 = 0.5 * c.mean() * conn ** 2

        if self.hyperparameter['alpha'] == 0:
            h_D = 0
            lambda_h_wD = 0
            c_hw_2 = 0
        else:
            h_D = h_W[self.constraint_type](G_D)
            lambda_h_wD = (ld * h_D).mean()  # lagrangian term
            c_hw_2 = 0.5 * (c * h_D * h_D).mean() # l2 penalty

        # group norm
        w_l1_l2 = torch.linalg.norm(G_D, ord=2, dim=0).sum()
        rho_w_l1 = self.rho * w_l1_l2 / X.shape[0] / (self.db.n / 10)

        loss = loss_se + lambda_conn + c_conn_2 + mu_one + rho_w_l1 + (lambda_h_wD + c_hw_2)

        log = {'SE': loss_se.item(),
               'l1/l2': w_l1_l2.item(),
               'h_D': h_D.mean().item() if self.hyperparameter['alpha'] > 0 else 0,
               'conn': conn.item(),
               'one': one.item()
               }
        return loss, conn.mean().item(), log

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
