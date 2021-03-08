import torch
import math
import numpy as np
from gan4dag.common.consts import DEVICE
import os
import pickle as pkl
from gan4dag.mmd_utils import MMD


class Eval:
    def __init__(self, database, save_dir, model_dump, save_itr, baseline=False):
        self.db = database

        data_hp = 'LSEM-d-%d-ts-%.2f-sp-%.2f' % (self.db.d, self.db.W_threshold, self.db.W_sparsity)
        self.save_dir = save_dir + '/' + data_hp
        if baseline:
            self.save_dir += '/baseline'
        self.model_dump = model_dump
        self.save_itr = save_itr

    def eval(self, g_net, m_small=100, m_large=500, verbose=True, bw=1.0):
        self.itr = self.save_itr
        result = {'mmd': [[], []], 'ce': [[], []], 'parameter': [[], []]}
        while True:
            dump = self.save_dir + '/Itr-%d-' % self.itr + self.model_dump
            if os.path.isfile(dump):
                g_net.load_state_dict(torch.load(dump))
                g_net.eval()
            else:
                print('Finished. File not exist: %s' % dump)
                filename = self.save_dir + '/' + self.model_dump[:-5] + '.result'
                with open(filename, 'wb') as f:
                    pkl.dump(result, f)
                return result

            # -----------------
            #  Evaluate MMD, CE, Parameter
            # -----------------
            # result['mmd'][0].append(self.itr)
            # result['mmd'][1].append(self.mmd(g_net, m_small, verbose, bw))

            result['ce'][0].append(self.itr)
            result['ce'][1].append(self.ce(g_net, m_large, verbose))

            result['parameter'][0].append(self.itr)
            result['parameter'][1].append(self.parameter_compare(g_net, verbose))

            self.itr += self.save_itr

    def mmd(self, g_net, m=100, verbose=True, bw=1.0):
        with torch.no_grad():
            W_gen, _, _ = g_net.gen_batch_dag(m, diff=False)

            key = 'test-dags-'+str(m)
            if key not in self.db.static:
                self.db.static[key] = self.db.gen_dags(m)
            W_real = torch.tensor(self.db.static[key]).to(DEVICE)

            mmd_val = MMD(W_gen.view(m, -1), W_real.view(m, -1), bandwidth=bw).item()
            if verbose:
                print('(%d) MMD: %.3f' % (self.itr, mmd_val))

            return mmd_val

    def ce(self, g_net, m=100, verbose=True):
        with torch.no_grad():
            key = 'test-z-' + str(m)
            if key not in self.db.static:
                self.db.static[key] = torch.normal(0, 1, size=(m, self.db.d, self.db.d)).float().to(DEVICE)

            # samples from generator
            W_gen = g_net(self.db.static[key])
            # probability in true distribution
            W_mean_true = torch.tensor(self.db.W_mean).to(DEVICE)
            W_sd_true = torch.tensor(self.db.W_sd).to(DEVICE)

            log_p_W_ij = -1/2 * ((W_gen - W_mean_true) / W_sd_true) ** 2 - W_sd_true * math.sqrt(2 * math.pi)
            log_p_W = torch.sum(torch.sum(log_p_W_ij, dim=-1), dim=-1)
            cross_entropy = - torch.mean(log_p_W).item()
        if verbose:
            print('(%d) CE: %.3f' % (self.itr, cross_entropy))
        return cross_entropy

    def parameter_compare(self, g_net, verbose=True):
        W_mean_true = self.db.W_mean
        W_sd_true = self.db.W_sd
        noise_mean_true = self.db.noise_mean
        noise_sd_true = self.db.noise_sd

        W_mean_err = np.sqrt(((W_mean_true - g_net.W.data.cpu().numpy()) ** 2).sum())
        W_sd_err = np.sqrt(((W_sd_true - g_net.V.data.cpu().numpy()) ** 2).sum())
        noise_mean_err = np.sqrt(((noise_mean_true - g_net.noise_mean.data.cpu().numpy())**2).sum())
        noise_sd_err = np.sqrt(((noise_sd_true - g_net.noise_sd.data.cpu().numpy())**2).sum())

        if verbose:
            print('(%d) Error: W_m: %.3f, W_s: %.3f, n_m: %.3f, n_d: %.3f' % (self.itr, W_mean_err, W_sd_err, noise_mean_err, noise_sd_err))
        return [W_mean_err, W_sd_err, noise_mean_err, noise_sd_err]

