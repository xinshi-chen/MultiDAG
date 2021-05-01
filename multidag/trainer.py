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
        default = {'rho': 0.2, 'lambda': 0.1, 'c': 1.0, 'eta': 0.01,
                   'nu': 0.2, 'threshold': 1e-1, 'dual_interval': 5}

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

    def train(self, epochs, batch_size=0, start_epoch=0, loss_type=None):
        """
        training logic
        """
        progress_bar = tqdm(range(start_epoch, start_epoch + epochs))
        dsc = ''
        X = self.db.load_data(device=DEVICE)
        print('*** Start training ***')
        for e in progress_bar:
            self._train_epoch(e, X, progress_bar, dsc, loss_type)
        return

    def _train_epoch(self, epoch, X, progress_bar, dsc, loss_type):

        self.g_dag.train()

        # -----------------
        #  primal step
        # -----------------
        self.optimizer.zero_grad()

        loss, h_D, log = self.get_loss(X, self.ld, self.c, phase='train')   # TODO
        loss.backward()

        self.optimizer.step()   # TODO

        # -----------------
        #  dual step
        # -----------------
        # TODO
        if (epoch + 1) % self.hyperparameter['dual_interval'] == 0:
            self.ld += (1 / self.db.d) * (10 - (10 - h_D) * ((10 - h_D) > 0))
            self.c = torch.clamp(self.c * (1 + self.hyperparameter['eta']), min=0, max=10)

        # TODO
        progress_bar.set_description("[SE: %.2f] [l1/l2: %.2f] [hw: %.2f] [conn: %.2f] [ld: %.2f, c: %.2f]" %
                                     (log['SE'], log['l1/l2'], log['h_D'], log['conn'],
                                      self.ld.mean().item(), self.c.mean().item()) + dsc)

            # TODO: save
        self.save(epoch)
        return

    def get_loss(self, X, ld, c, phase='train'):
        # TODO
        G_D = self.g_dag.G * self.g_dag.T
        # Squared Error
        loss_se = LSEM.SE(G_D, X)

        # dagness constraints
        conn = h_W[self.constraint_type](self.g_dag.T)
        # entropy = - self.hyperparameter['nu'] * (self.g_dag.T * torch.log(self.g_dag.T + 1e-20) + (
        #         1 -  self.g_dag.T) * torch.log(1 - self.g_dag.T + 1e-20)).sum()
        h_D = h_W[self.constraint_type](G_D)
        # h_D = h_W[self.constraint_type](self.g_dag.T)  # scalar
        lambda_h_wD = (ld * h_D).sum()  # lagrangian term
        c_hw_2 = 0.5 * (c * h_D * h_D).sum()  # l2 penalty

        # group norm
        # w_l1_l2 = torch.linalg.norm(self.g_dag.G, ord=2, dim=0).sum()
        w_l1_l2 = torch.linalg.norm(G_D, ord=2, dim=0).sum()
        rho_w_l1 = self.hyperparameter['rho'] * w_l1_l2

        loss = loss_se + self.hyperparameter['nu'] * conn + rho_w_l1 + lambda_h_wD + c_hw_2

        log = {'SE': loss_se.item(),
               'l1/l2': w_l1_l2.item(),
               'h_D': h_D.sum().item(),
               'conn': conn.item()
               }
        return loss, h_D.sum().item(), log

    def save(self, epoch):
        dump = os.path.join(self.save_dir, self.model_dump + f'_{epoch}')
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
            for key in log:
                print(f'{key}:\t {log[key]:.3f}')


