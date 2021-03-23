import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from vae4dag.common.consts import DEVICE
from vae4dag.eval import Eval
from vae4dag.dag_utils import h_W
import os
import math


D_Loss = torch.nn.BCEWithLogitsLoss(reduction='mean')


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
        default = {'rho': 0.1, 'alpha': 1.0, 'lambda': 0.1, 'c': 1.0, 'p': 0.5}
        for key in default:
            if key in hyperparameters:
                self.hyperparameter[key] = hyperparameters[key]
            else:
                self.hyperparameter[key] = default[key]

        self.n = data_base.n
        self.k = math.floor(self.n * self.hyperparameter['p'])

        # initialize lambda and c
        self.ld = torch.ones(size=[self.db.num_dags['train']]).to(DEVICE) * self.hyperparameter['lambda']
        self.c = torch.ones(size=[self.db.num_dags['train']]).to(DEVICE) * self.hyperparameter['c']
        self.hw_prev = torch.ones(size=[self.db.num_dags['train']]).to(DEVICE) * math.inf

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
        self.decoder.train()
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

            m = X.shape[0]  # number of DAGs in this batch

            self.e_optimizer.zero_grad()
            self.d_optimizer.zero_grad()

            # compute W
            W = self.encoder(X_in)   # [m, d, d]

            # compute neg log likelihood
            nll_eval = torch.mean(torch.sum(self.decoder.NLL(W, X_eval), -1))  # [1]

            # dagness loss
            hw = h_W[self.constraint_type](W)  # [m]
            if epoch >= 1: #10:
                with torch.no_grad():
                    hw_new = hw.data
                    self.update_lambda_c(hw_new, idx)
                    self.hw_prev[idx] = hw_new

            lambda_hw = torch.mean(self.ld[idx].detach() * hw)  # (hw - 0.05 * w_l1))

            # dagness - l2 penalty
            c_hw_2 = torch.mean(0.5 * self.c[idx].detach() * (hw * hw))  # - 0.05 * w_l1))

            # l1 regularization
            w_l1 = torch.sum(torch.abs(W).view(m, -1), dim=-1)  #[m]
            w_l1 = w_l1.mean()

            loss = nll_eval + self.hyperparameter['rho'] * w_l1 #+ lambda_hw + c_hw_2

            # backward
            loss.backward()
            self.e_optimizer.step()
            self.d_optimizer.step()

            progress_bar.set_description("[Epoch %.2f] [nll: %.3f / %.3f / %.3f] [l1: %.2f] [hw: %.2f] [ld: %.2f, c: %.2f]" %
                                         (epoch + float(it + 1) / num_iterations, nll_eval.item(), true_nll_eval.mean(), self.best_vali_nll,
                                          w_l1.item(), hw.mean().item(), self.ld.mean().item(), self.c.mean().item()) + dsc)

            # -----------------
            #  Validation & Save
            # -----------------
            self.train_itr += 1
            last_itr = (self.train_itr == tot_epoch * num_iterations)
            if self.train_itr % self.save_itr == 0:
                nll_vali, hw_vali = self.valiation(self.k, hw_tol=math.inf)
                if nll_vali is not None:
                    if nll_vali < self.best_vali_nll:
                        self.best_vali_nll = nll_vali
                        self.save(self.train_itr, best=True)
            if last_itr:
                self.save(self.train_itr, best=False)
        return

    def update_lambda_c(self, hw_new, idx):
        # update lambda and c
        with torch.no_grad():
            self.ld[idx] += 0 # (1 / (self.db.d)) * (10 - F.relu(10 - hw_new))   # 1/d * max(hw, 10)
            gamma_hw_old = self.hyperparameter['gamma'] * self.hw_prev[idx]
            self.c[idx] += 0 # (self.hyperparameter['eta'] * self.c[idx]) * (hw_new > gamma_hw_old).float()

    def valiation(self, k, hw_tol):
        self.encoder.eval()
        self.decoder.eval()

        with torch.no_grad():
            X, nll = self.db.static_data['vali']
            X_in, true_nll_in = X[:, :k, :], nll[:, :k]
            X_eval, true_nll_eval = X[:, k:, :], nll[:, k:]

            W = self.encoder(X_in.to(DEVICE))
            hw = h_W[self.constraint_type](W).mean().item()
            if 1:
                W = Eval.project_W(W, DEVICE, verbose=False, sparsity=1.0, max_itr=5)
                nll_eval = torch.sum(self.decoder.NLL(W, X_eval.to(DEVICE)), dim=-1)  # [m, n-k]
                nll_eval = nll_eval.mean().item()
            else:
                nll_eval = None
        return nll_eval, hw

    def save(self, itr, best):

        if best:
            key = '/best-'
        else:
            key = '/Itr-%d-' % itr
        dump = self.save_dir + key + self.model_dump
        torch.save(self.encoder.state_dict(), dump)

        dump = dump[:-5] + '_decoder.dump'
        torch.save(self.decoder.state_dict(), dump)

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
                loss_mse = ((W_est - W_batch.detach()) ** 2).view(M, -1).sum(dim=-1).mean()

                # dagness loss
                h_wD = h_W[self.constraint_type](self.w_dag(idx))  # [m]

                lambda_h_wD = (self.ld[idx].detach() * h_wD).mean()
                c_hw_2 = 0.5 * self.hyperparameter['c'] * (h_wD * h_wD).mean()  # dagness - l2 penalty

                # l1 regularization
                m = W_est.shape[0]
                w_l1 = torch.sum(torch.abs(W_est).view(m, -1), dim=-1).mean()  #[m]
                rho_w_l1 = self.hyperparameter['rho'] * w_l1

                # ||w_dag - w||
                w_dist = ((self.w_dag(idx) - W_est) ** 2).view(m, -1).sum(-1).mean()
                alpha_w_dist = self.hyperparameter['alpha'] / (2 * self.db.d) * w_dist

                loss = loss_mse + rho_w_l1 + lambda_h_wD + c_hw_2 + alpha_w_dist

                loss.backward()
                self.e_optimizer.step()
                self.w_optimizer.step()

                # validation

                it += 1
                itr += 1

                progress_bar.set_description("[Epoch %.2f] [loss: %.3f / %.3f] [w_dis: %.2f] [l1: %.2f] [hwD: %.2f] [ld: %.2f]" %
                                         (epoch + float(it + 1) / num_iterations, loss_mse.item(), best_vali_loss,
                                          w_dist.item(), w_l1.item(), h_wD.mean().item(), self.ld.mean().item()))
