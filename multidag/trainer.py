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
    def __init__(self, g_dag, optimizer, data_base, save_dir, model_dump, constraint_type='notears', hyperparameters={}):

        self.g_dag = g_dag
        self.db = data_base

        self.optimizer = optimizer
        self.train_itr = 0
        self.constraint_type = constraint_type

        self.model_dump = model_dump

        # TODO hyperparameters may be different
        self.hyperparameter = dict()
        default = {'rho': 0.2, 'lambda': 1.0, 'c': 1.0, 'eta': 0.1, 'mu': 0.1, 'threshold': 1e-1, 'dual_interval': 5}

        for key in default:
            if key in hyperparameters:
                self.hyperparameter[key] = hyperparameters[key]
            else:
                self.hyperparameter[key] = default[key]

        self.n = data_base.n

        # initialize lambda and alpha
        self.ld = torch.ones(size=[self.db.K]).to(DEVICE) * self.hyperparameter['lambda']
        self.c = torch.ones(size=[self.db.K]).to(DEVICE) * self.hyperparameter['c']

        # make the save_dir with hyperparameter index
        self.save_dir = save_dir + '/' + data_base.hp
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

    def train(self, epochs, start_epoch=0, loss_type=None):
        """
        training logic
        """
        progress_bar = tqdm(range(start_epoch, start_epoch + epochs))
        dsc = ''
        X = self.db.load_data(device=DEVICE)
        print('*** Start training ***')
        for e in progress_bar:
            self._train_epoch(e, X, progress_bar, dsc, loss_type)
        self.save()
        return

    def _train_epoch(self, epoch, X, progress_bar, dsc, loss_type):

        self.g_dag.train()

        # -----------------
        #  primal step
        # -----------------
        self.optimizer.zero_grad()

        loss, h_D, log = self.get_loss(X, self.ld, self.c)
        loss.backward()

        self.optimizer.step()

        # -----------------
        #  dual step
        # -----------------
        if (epoch + 1) % self.hyperparameter['dual_interval'] == 0:
            self.ld += (20 - (20 - h_D) * ((20 - h_D) > 0))
            self.c = torch.clamp(self.c * (1 + self.hyperparameter['eta']), min=0, max=20)

        # info
        progress_bar.set_description("[SE: %.2f] [l1/l2: %.2f] [hw: %.2f] [conn: %.2f, one: %.2f] [ld: %.2f, c: %.2f]" %
                                     (log['SE'], log['l1/l2'], log['h_D'], log['conn'], log['one'],
                                      self.ld.mean().item(), self.c.mean().item()) + dsc)
        return

    def get_loss(self, X, ld, c):
        G_D = self.g_dag.G * self.g_dag.T
        # Squared Error
        loss_se = LSEM.SE(G_D, X)

        # dagness constraints
        one = (self.g_dag.T).abs().mean()
        mu_one = self.hyperparameter['mu'] * one

        conn = h_W[self.constraint_type](self.g_dag.T)
        lambda_conn = ld.mean() * conn
        c_conn_2 = 0.5 * c.mean() * conn ** 2

        h_D = h_W[self.constraint_type](G_D)
        lambda_h_wD = (ld * h_D).mean()  # lagrangian term
        c_hw_2 = 0.5 * (c * h_D * h_D).mean()  # l2 penalty

        # group norm
        w_l1_l2 = torch.linalg.norm(G_D, ord=2, dim=0).sum()
        rho_w_l1 = self.hyperparameter['rho'] * w_l1_l2

        loss = loss_se + lambda_conn + c_conn_2 + mu_one + rho_w_l1 + lambda_h_wD + c_hw_2

        log = {'SE': loss_se.item(),
               'l1/l2': w_l1_l2.item(),
               'h_D': h_D.sum().item(),
               'conn': conn.item(),
               'one': one.item()
               }
        return loss, h_D.sum().item(), log

    def save(self):
        dump = os.path.join(self.save_dir, self.model_dump)
        torch.save(self.g_dag.state_dict(), dump)

    def evaluate(self):
        G_true = np.abs(np.sign(self.db.G))
        G_est = np.abs((self.g_dag.G * self.g_dag.T).detach().cpu().numpy())
        T = self.g_dag.T.squeeze().detach().cpu().numpy()
        T[T < self.hyperparameter['threshold']] = 0
        G_est[G_est < self.hyperparameter['threshold']] = 0
        G_est = np.sign(G_est)
        for k in range(G_true.shape[0]):
            print(f'##### result for graph {k} #####')
            log = count_accuracy(G_true[k], G_est[k])
            print(log)


