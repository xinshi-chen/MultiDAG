import torch
import pdb
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from vae4dag.common.consts import DEVICE, OPTIMIZER
from vae4dag.common.cmd_args import cmd_args
from vae4dag.eval import Eval
from vae4dag.model import W_DAG
from vae4dag.dag_utils import h_W
import os
import math


D_Loss = torch.nn.BCEWithLogitsLoss(reduction='mean')


def MSE(w1, w2):
    assert len(w1.shape) == 3
    assert len(w2.shape) == 3
    m = w1.shape[0]
    return ((w1 - w2) ** 2).view(m, -1).sum(dim=-1).mean()


class Trainer:
    def __init__(self, encoder, decoder, w_dag, e_optimizer, d_optimizer, w_optimizer, data_base, save_dir, model_dump,
                 save_itr=500, constraint_type='notears', hyperparameters={}):
        self.encoder = encoder
        self.decoder = decoder
        self.w_dag = w_dag
        self.db = data_base
        self.e_optimizer = e_optimizer
        self.d_optimizer = d_optimizer
        self.w_optimizer = w_optimizer
        self.train_itr = 0
        self.save_itr = save_itr
        self.constraint_type = constraint_type

        self.model_dump = model_dump

        self.hyperparameter = dict()
        default = {'rho': 0.1, 'alpha': 1.0, 'lambda': 0.1, 'c': 1.0, 'p': 0.5, 'eta': 0.01}
        for key in default:
            if key in hyperparameters:
                self.hyperparameter[key] = hyperparameters[key]
            else:
                self.hyperparameter[key] = default[key]

        self.n = data_base.n
        self.k = math.floor(self.n * self.hyperparameter['p'])

        # initialize lambda and alpha
        self.ld = torch.ones(size=[self.db.num_dags['train']]).to(DEVICE) * self.hyperparameter['lambda']
        self.alpha = self.hyperparameter['alpha']

        # make the save_dir with hyperparameter index
        self.save_dir = save_dir + '/' + data_base.dataset_hp
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        train_hp = []
        for key in default:
            train_hp.append(key)
            train_hp.append(str(self.hyperparameter[key]))
        train_hp.append(constraint_type)

        self.train_hp = "-".join(train_hp)
        self.save_dir += '/' + self.train_hp
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

        self.best_vali_nll = math.inf

    def train(self, epochs, batch_size, start_epoch=0):
        """
        training logic
        """

        if start_epoch > 0:
            completed_epochs = start_epoch
            num_itr_per_epoch = len(range(0, self.db.num_dags['train'], batch_size))
            itr = num_itr_per_epoch * completed_epochs
            self.load(itr)

        progress_bar = tqdm(range(start_epoch, start_epoch + epochs))
        dsc = ''
        print('*** Start training ***')
        for e in progress_bar:
            self._train_epoch(e, epochs, batch_size, progress_bar, dsc)

        return

    def _train_epoch(self, epoch, tot_epoch, batch_size, progress_bar, dsc):
        self.encoder.train()
        if self.d_optimizer is not None:
            self.decoder.train()
        self.w_dag.train()

        data_loader = self.db.load_data(batch_size=batch_size,
                                        auto_reset=False,
                                        shuffle=True,
                                        phase='train',
                                        device=DEVICE)
        num_iterations = len(range(0, self.db.num_dags['train'], batch_size))

        for it, data in enumerate(data_loader):
            X, idx, true_nll = data
            X, true_nll = self.db.shuffle_order_of_sample(X, true_nll, device=DEVICE)

            X_in, X_eval = X[:, :self.k, :].detach(), X[:, self.k:, :].detach()
            true_nll_in, true_nll_eval = true_nll[:, :self.k], true_nll[:, self.k:]

            self.e_optimizer.zero_grad()
            if self.d_optimizer is not None:
                self.d_optimizer.zero_grad()
            self.w_optimizer.zero_grad()

            loss, h_wD, log = self.get_loss(X_in=X_in.detach(), X_eval=X_eval.detach(), W_D=self.w_dag(idx),
                                            ld=self.ld[idx].detach(), w_dist=False)
            loss.backward()

            # -----------------
            #  primal step
            # -----------------
            self.e_optimizer.step()
            if self.d_optimizer is not None:
                self.d_optimizer.step()
            self.w_optimizer.step()

            # -----------------
            #  dual step
            # -----------------
            self.ld[idx] += (1 / self.db.d) * (10 - F.relu(10 - h_wD.detach()))
            # update alpha
            self.alpha = min(10, self.alpha * (1 + self.hyperparameter['eta']))

            progress_bar.set_description("[Epoch %.2f] [nll: %.3f / %.3f / %.3f] [w_dis: %.2f] [l1: %.2f] [hw: %.2f] [ld: %.2f, ap: %.2f]" %
                                         (epoch + float(it + 1) / num_iterations, log['nll'], true_nll_eval.mean(), self.best_vali_nll,
                                          log['w_dist'], log['l1'], log['hw'], self.ld.mean().item(), self.alpha) + dsc)

            # -----------------
            #  Validation & Save
            # -----------------
            self.train_itr += 1
            last_itr = (self.train_itr == tot_epoch * num_iterations)
            if self.train_itr % self.save_itr == 0:
                nll_vali = self.valiation()
                
                self.encoder.train()
                if self.d_optimizer is not None:
                    self.decoder.train()

                if nll_vali < self.best_vali_nll:
                    self.best_vali_nll = nll_vali
                    self.save(self.train_itr, best=True)
            if last_itr:
                self.save(self.train_itr, best=False)
        return

    def get_loss(self, X_in, X_eval, W_D, ld, phase='train', w_dist=True):

        # neg-log-likelihood
        if phase == 'train':
            loss_nll = self.decoder.NLL(W_D, X_eval).sum(dim=-1).mean()
        else:
            loss_nll = torch.tensor(0.0)

        # distance ||W_D - W_est||
        if w_dist:
            W_est = self.encoder(X_in)
            if phase != 'train':
                W_est = W_est.detach()
            w_dist = MSE(W_D, W_est)
            alpha_w_dist = self.alpha / (2 * self.db.d) * w_dist
        else:
            w_dist = torch.tensor(0.0)
            alpha_w_dist = w_dist.to(DEVICE)

        # dagness constraints
        h_wD = h_W[self.constraint_type](W_D)  # [m]
        lambda_h_wD = (ld * h_wD).mean()  # lagrangian term
        c_hw_2 = 0.5 * self.hyperparameter['c'] * (h_wD * h_wD).mean()  # l2 penalty

        # l1 regularization
        w_l1 = torch.sum(torch.abs(W_D).view(W_D.shape[0], -1), dim=-1).mean()
        rho_w_l1 = self.hyperparameter['rho'] * w_l1

        loss = loss_nll + rho_w_l1 + lambda_h_wD + c_hw_2 + alpha_w_dist

        log = {'nll': loss_nll.item(),
               'l1': w_l1.item(),
               'hw': h_wD.mean().item(),
               'w_dist': w_dist.item()
               }
        return loss, h_wD, log

    def valiation(self, num_itr=1000):
        self.encoder.eval()
        if self.d_optimizer is not None:
            self.decoder.eval()

        X, true_nll = self.db.static_data['vali']
        X_in, X_eval = X[:, :self.k, :].detach().to(DEVICE), X[:, self.k:, :].detach().to(DEVICE)

        # -----------------
        #  Project W_est to DAG by optimizing over W_D
        # -----------------
        progress_bar = tqdm(range(0, num_itr))
        w_dag = W_DAG(num_dags=self.db.num_dags['vali'], d=self.db.d).to(DEVICE)
        optimizer = OPTIMIZER[cmd_args.w_optimizer](w_dag.parameters(), lr=cmd_args.w_lr,
                                                    weight_decay=cmd_args.weight_decay)
        ld = torch.ones(size=[self.db.num_dags['vali']]).to(DEVICE) * self.hyperparameter['lambda']
        alpha = self.alpha

        # init W_D as W_est
        w_dag.w.data = self.encoder(X_in).detach()

        for it in progress_bar:
            optimizer.zero_grad()
            loss, h_wD, log = self.get_loss(X_in=X_in.detach(), X_eval=X_eval.detach, W_D=w_dag.w, ld=ld.detach(), phase='vali')
            loss.backward()
            optimizer.step()
            # update lambda
            ld += (1 / self.db.d) * (10 - F.relu(10 - h_wD))
            # update alpha
            alpha = min(10, alpha * (1 + self.hyperparameter['eta']))
            progress_bar.set_description("[itr %.2f] [w_dis: %.2f] [l1: %.2f] [hwD: %.2f] [ld: %.2f] [af: %.2f]" %
                                         (it, log['w_dist'], log['l1'], log['hw'], ld.mean().item(), alpha))
        # -----------------
        #  Evaluate neg-log-likelihood on projected DAG
        # -----------------
        with torch.no_grad():
            nll_eval = torch.sum(self.decoder.NLL(w_dag.w, X_eval), dim=-1).mean().item()  # [m, n-k]

        return nll_eval

    def save(self, itr, best):

        if best:
            key = '/best-'
        else:
            key = '/Itr-%d-' % itr
        # encoder
        dump = self.save_dir + key + self.model_dump
        torch.save(self.encoder.state_dict(), dump)
        # decoder
        if self.d_optimizer is not None:
            dump = dump[:-5] + '_decoder.dump'
            torch.save(self.decoder.state_dict(), dump)
        # current w_d
        dump = dump[:-5] + '_wD.dump'
        torch.save(self.w_dag.state_dict(), dump)

    def load(self, itr):

        dump = self.save_dir + '/Itr-%d-' % itr + self.model_dump
        self.encoder.load_state_dict(torch.load(dump))

        dump = dump[:-5] + '_decoder.dump'
        self.decoder.load_state_dict(torch.load(dump))

    def train_with_W(self, X, W, X_vali, W_vali, epochs, batch_size):
        self.encoder.train()
        self.w_dag.train()

        if isinstance(W, np.ndarray):
            W = torch.tensor(W)
        if isinstance(W_vali, np.ndarray):
            W_vali = torch.tensor(W_vali)

        W = W.to(DEVICE)
        X = X.to(DEVICE)
        W_vali = W_vali.to(DEVICE)
        X_vali = X_vali.to(DEVICE)
        index = torch.arange(0, X.shape[0]).to(DEVICE)

        M = W.shape[0]

        progress_bar = tqdm(range(0, epochs))
        num_iterations = len(range(0, M, batch_size))

        itr = 0
        best_vali_loss = math.inf
        for epoch in progress_bar:
            perms = torch.randperm(M)
            W = W[perms]
            X = X[perms]
            index = index[perms]
            it = 0
            for pos in range(0, M, batch_size):

                num_w = min(batch_size, M-pos)
                W_batch = W[pos: pos+num_w]
                X_batch = X[pos: pos+num_w]
                idx = index[pos: pos+num_w]
                self.e_optimizer.zero_grad()
                self.w_optimizer.zero_grad()

                # loss
                W_est = self.encoder(X_batch.detach())
                loss_mse = MSE(self.w_dag(idx), W_batch.detach())

                # ||w_dag - w||
                w_dist = MSE(self.w_dag(idx), W_est)
                alpha_w_dist = self.hyperparameter['alpha'] / (2 * self.db.d) * w_dist

                # dagness loss
                h_wD = h_W[self.constraint_type](self.w_dag(idx))  # [m]

                lambda_h_wD = (self.ld[idx].detach() * h_wD).mean()
                c_hw_2 = 0.5 * self.hyperparameter['c'] * (h_wD * h_wD).mean()  # dagness - l2 penalty

                # l1 regularization
                m = W_est.shape[0]
                # w_l1 = torch.sum(torch.abs(W_est).view(m, -1), dim=-1).mean()  #[m]
                w_l1 = torch.sum(torch.abs(self.w_dag(idx)).view(m, -1), dim=-1).mean()  #[m]
                rho_w_l1 = self.hyperparameter['rho'] * w_l1

                loss = loss_mse + rho_w_l1 + lambda_h_wD + c_hw_2 + alpha_w_dist

                loss.backward()
                self.e_optimizer.step()
                self.w_optimizer.step()

                # update lambda
                self.ld[idx] += (1 / self.db.d) * (10 - F.relu(10 - h_wD))
                # update alpha
                self.hyperparameter['alpha'] = min(10, self.hyperparameter['alpha'] * (1 + self.hyperparameter['eta']))
                itr += 1

                # validation
                if itr % self.save_itr == 0:
                    vali_loss = self.train_with_W_validation(X_vali, W_vali)
                    if best_vali_loss > vali_loss:
                        best_vali_loss = vali_loss
                it += 1

                progress_bar.set_description("[Epoch %.2f] [loss: %.3f / %.3f] [w_dis: %.2f] [l1: %.2f] [hwD: %.2f] [ld: %.2f]" %
                                         (epoch + float(it + 1) / num_iterations, loss_mse.item(), best_vali_loss,
                                          w_dist.item(), w_l1.item(), h_wD.mean().item(), self.ld.mean().item()))

    def train_with_W_validation(self, X_vali, W_vali, num_itr=1000):
        progress_bar = tqdm(range(0, num_itr))
        w_dag = W_DAG(num_dags=self.db.num_dags['vali'], d=self.db.d).to(DEVICE)
        optimizer = OPTIMIZER[cmd_args.w_optimizer](w_dag.parameters(), lr=cmd_args.w_lr,
                                                    weight_decay=cmd_args.weight_decay)
        ld = torch.ones(size=[self.db.num_dags['vali']]).to(DEVICE) * self.hyperparameter['lambda']
        alpha = self.hyperparameter['alpha']
        m = X_vali.shape[0]
        for it in progress_bar:
            optimizer.zero_grad()
            loss_mse = 0.0 # MSE(w_dag.w, W_vali)
            # ||w_dag - w||
            w_est = self.encoder(X_vali).detach()
            w_dist = MSE(w_dag.w, w_est)
            # alpha_w_dist = alpha / (2 * self.db.d) * w_dist
            h_wD = h_W[self.constraint_type](w_dag.w)  # [m]
            lambda_h_wD = (ld.detach() * h_wD).mean()
            c_hw_2 = 0.5 * self.hyperparameter['c'] * (h_wD * h_wD).mean()  # dagness - l2 penalty
            w_l1 = torch.sum(torch.abs(w_dag.w).view(m, -1), dim=-1).mean()  #[m]
            rho_w_l1 = self.hyperparameter['rho'] * w_l1

            loss = loss_mse + rho_w_l1 + lambda_h_wD + c_hw_2 + w_dist # alpha_w_dist

            loss.backward()

            optimizer.step()
            # update lambda
            ld += (1 / self.db.d) * (10 - F.relu(10 - h_wD))
            # update alpha
            alpha = min(10, alpha * (1 + self.hyperparameter['eta']))
            progress_bar.set_description("[itr %.2f] [loss: %.3f] [w_dis: %.2f] [l1: %.2f] [hwD: %.2f] [ld: %.2f] [af: %.2f]" %
                                         (it, loss_mse, w_dist.item(), w_l1.item(), h_wD.mean().item(),
                                          ld.mean().item(), alpha))
        loss_mse = MSE(w_dag.w, W_vali)
        return loss_mse.item()
