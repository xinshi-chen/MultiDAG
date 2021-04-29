import torch
import torch.nn.functional as F
from tqdm import tqdm
from multidag.common.consts import DEVICE, OPTIMIZER
from multidag.dag_utils import h_W
import os


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
        default = {'rho': 0.1, 'alpha': 1.0, 'lambda': 0.1, 'c': 1.0, 'p': 0.5, 'eta': 0.01}

        for key in default:
            if key in hyperparameters:
                self.hyperparameter[key] = hyperparameters[key]
            else:
                self.hyperparameter[key] = default[key]

        self.n = data_base.n

        # initialize lambda and alpha
        self.ld = torch.ones(size=[self.db.num_dags['train']]).to(DEVICE) * self.hyperparameter['lambda']
        self.alpha = self.hyperparameter['alpha']

        # make the save_dir with hyperparameter index
        self.save_dir = save_dir + '/' + data_base.hp
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

    def train(self, epochs, batch_size, start_epoch=0, loss_type=None):
        """
        training logic
        """
        progress_bar = tqdm(range(start_epoch, start_epoch + epochs))
        dsc = ''
        print('*** Start training ***')
        for e in progress_bar:
            self._train_epoch(e, epochs, batch_size, progress_bar, dsc, loss_type)

        return

    def _train_epoch(self, epoch, tot_epoch, batch_size, progress_bar, dsc, loss_type):

        self.g_dag.train()

        data_loader = self.db.load_data(batch_size=batch_size,
                                        auto_reset=False,
                                        shuffle=True,
                                        phase='train',
                                        device=DEVICE)

        num_iterations = len(range(0, self.db.K, batch_size))

        for it, data in enumerate(data_loader):
            X, idx = data

            self.optimizer.zero_grad()

            loss, h_wD, log = self.get_loss()   # TODO
            loss.backward()

            # -----------------
            #  primal step
            # -----------------

            self.optimizer.step()   # TODO

            # -----------------
            #  dual step
            # -----------------
            # TODO
            self.ld[idx] += (1 / self.db.d) * (10 - F.relu(10 - h_wD))
            # update alpha
            self.alpha = min(10, self.alpha * (1 + self.hyperparameter['eta']))

            # TODO
            progress_bar.set_description("[Epoch %.2f] [nll: %.3f / %.3f / %.3f] [w_dis: %.2f] [l1: %.2f] [hw: %.2f] [ld: %.2f, ap: %.2f]" %
                                         (epoch + float(it + 1) / num_iterations, log['nll'], true_nll_eval.mean(), self.best_vali_nll,
                                          log['w_dist'], log['l1'], log['hw'], self.ld.mean().item(), self.alpha) + dsc)

            # TODO: save
        return

    def get_loss(self, X, G_D, ld, phase='train'):
        # TODO

        # Squared Error
        loss_se = 0
        # TODO: use LSEM.SE in model.py

        # dagness constraints
        h_wD = h_W[self.constraint_type](G_D)  # [m]
        lambda_h_wD = (ld * h_wD).mean()  # lagrangian term
        c_hw_2 = 0.5 * self.hyperparameter['c'] * (h_wD * h_wD).mean()  # l2 penalty

        # group norm
        # TODO: change the current l1 norm to group norm
        w_l1_l2 = torch.sum(torch.abs(G_D).view(G_D.shape[0], -1), dim=-1).mean()
        rho_w_l1 = self.hyperparameter['rho'] * w_l1_l2

        loss = loss_se + rho_w_l1 + lambda_h_wD + c_hw_2

        log = {'SE': loss_se.item(),
               'l1/l2': w_l1_l2.item(),
               'hw': h_wD.mean().item()
               }
        return loss, h_wD.detach(), log

    def save(self):
        dump = self.save_dir + self.model_dump
        torch.save(self.g_dag.state_dict(), dump)
