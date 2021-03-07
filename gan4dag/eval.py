import torch
import math
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from gan4dag.common.consts import DEVICE
import os
import pickle as pkl


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


def pairwise_distances(x, y=None):
    """
    Reference: https://github.com/xinshi-chen/ParticleFlowBayesRule/blob/2fb400f7a9ffe03cd654fe78f1d51c405cf6b7df/pfbayes/common/torch_utils.py#L21
    """
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x**2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y**2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)


def get_gamma(X, bandwidth):
    """
    Reference: https://github.com/xinshi-chen/ParticleFlowBayesRule/blob/master/pfbayes/common/distributions.py#L52
    """
    with torch.no_grad():
        x_norm = torch.sum(X ** 2, dim=1, keepdim=True)
        x_t = torch.transpose(X, 0, 1)
        x_norm_t = x_norm.view(1, -1)
        t = x_norm + x_norm_t - 2.0 * torch.matmul(X, x_t)
        dist2 = F.relu(Variable(t)).detach().data

        d = dist2.cpu().numpy()
        d = d[np.isfinite(d)]
        d = d[d > 0]
        median_dist2 = float(np.median(d))
        gamma = 0.5 / median_dist2 / bandwidth
        return gamma


def get_gamma_batch(X, bandwidth):

    with torch.no_grad():
        # X [m, n, d]
        m, n = X.shape[0], X.shape[1]
        x_norm = torch.sum(X ** 2, dim=2, keepdim=True)  # [m, n, 1]
        x_t = torch.transpose(X, 1, 2)  # [m, d, n]
        x_norm_t = torch.transpose(x_norm, 1, 2)   # [m, 1, n]
        t = x_norm + x_norm_t - 2.0 * torch.einsum('bij,bjk->bik', X, x_t)   # [m, n, n]
        dist2 = F.relu(Variable(t)).detach()
        dist2 = dist2.view(m, n*n)
        median_dist2, _ = torch.median(dist2, dim=-1)
        gamma = 0.5 / median_dist2 / bandwidth
        return gamma


def MMD_batch(x, y, bandwidth=1.0):
    # y [m, n, d]
    # x [m2, n2, d]

    # return mmd [m2, m]

    m2, n2 = x.shape[0], x.shape[1]
    m, n = y.shape[0], y.shape[1]

    y = y.detach()

    gamma = get_gamma_batch(y, bandwidth).detach()  # [m]
    kxx = kxx_batch(x, gamma)   # [m2, n2, n2]
    kxx = torch.sum(kxx.view(m2, -1), dim=-1) / n2 / (n2 - 1)  # [m2]

    kyy = kxx_batch(y, gamma)   # [m, n, n]
    kyy = (torch.sum(kyy.view(m, -1), dim=-1) / n / (n - 1)).detach()  # [m]

    kxy = kxy_batch(x, y, gamma)  # [m, m2, n, n2]
    kxy = torch.sum(kxy.view(m, m2, -1), dim=-1) / n / n2  # [m, m2]

    mmd = kxx.view(m2, 1) + kyy.view(1, m).detach() - 2 * kxy.transpose(0, 1)
    return mmd


def kxx_batch(x, gamma):
    """
    x [m, n, d]
    gamma [m]
    :return [m, n, n]
    """
    m = x.shape[0]
    n = x.shape[1]
    d = pairwise_distances_batch(x, x)  # [m, n, n]
    d = d * (-gamma).view(m, 1, 1).repeat(1, n, 1)  # [m, n, n]
    k = torch.exp(d)

    # make diagonal zero
    k = k * (1 - torch.eye(n).to(DEVICE).view(1, n, n).repeat(m, 1, 1))

    return k


def kxy_batch(x, y, gamma):
    """
    x [m2, n2, d]
    y [m, n, d]
    gamma [m]
    :return [m, n, n]
    """
    m, n = y.shape[0], y.shape[1]
    assert m == gamma.shape[0]

    m2, n2 = x.shape[0], x.shape[1]
    d = pairwise_distances_batch(y, x)  # [m, m2, n, n2]
    d = d * (-gamma).view(m, 1, 1, 1).repeat(1, m2, n, 1)  # [m, m2, n, n2]
    k = torch.exp(d)
    return k


def pairwise_distances_batch(x, y=None):
    '''
    Input: x is a [m, n, d] matrix
           y is an optional [m2, n2, d] matrix
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    m = x.shape[0]
    x_norm = (x**2).sum(2).reshape(m, -1, 1)  # [m, n, 1]
    if y is None:
        x_norm_t = torch.transpose(x_norm, 1, 2)  # [m, 1, n]
        x_t = torch.transpose(x, 1, 2)  # [m, d, n]
        dist = x_norm + x_norm_t - 2.0 * torch.einsum('bij,bjk->bik', x, x_t)
    else:
        m2 = y.shape[0]
        y_norm = (y**2).sum(2)  # [m2, n2]
        x_y_norm_sum = element_sum_batch(x_norm.view(m, -1), y_norm)  # [m, m2, n, n2]
        dist = x_y_norm_sum - 2.0 * torch.einsum('mnd,bkd->mbnk', x, y)  # [m, m2, n, n2]

    return torch.clamp(dist, 0.0, np.inf)


def element_sum_batch(x, y):
    '''
    :param x: [m, n]
    :param y: [m2, n2]
    :return: [m, m2, n, n2]
    '''
    m, n = x.shape
    m2, n2 = y.shape
    element_wise_sum = x.view(-1, 1) + y.view(1, -1)  # [m * n, m2 * n2]
    element_wise_sum = element_wise_sum.view(m, n, m2 * n2)
    element_wise_sum = element_wise_sum.view(m, n, m2, n2)
    element_wise_sum = element_wise_sum.transpose(1, 2)  # [m, m2, n, n2]

    return element_wise_sum


def get_kernel_mat(x, landmarks, gamma):
    """
    Reference: https://github.com/xinshi-chen/ParticleFlowBayesRule/blob/master/pfbayes/common/distributions.py#L68
    """
    d = pairwise_distances(x, landmarks)
    k = torch.exp(d * -gamma)
    k = k.view(x.shape[0], -1)
    return k


def MMD(x, y, bandwidth=1.0):
    """
    Reference: https://github.com/xinshi-chen/ParticleFlowBayesRule/blob/master/pfbayes/common/distributions.py#L75
    """
    y = y.detach()
    gamma = get_gamma(y.detach(), bandwidth)
    kxx = get_kernel_mat(x, x, gamma)
    idx = torch.arange(0, x.shape[0], out=torch.LongTensor())
    kxx = kxx * (1 - torch.eye(x.shape[0]).to(DEVICE))
    kxx = torch.sum(kxx) / x.shape[0] / (x.shape[0] - 1)

    kyy = get_kernel_mat(y, y, gamma)
    idx = torch.arange(0, y.shape[0], out=torch.LongTensor())
    kyy[idx, idx] = 0.0
    kyy = torch.sum(kyy) / y.shape[0] / (y.shape[0] - 1)
    kxy = torch.sum(get_kernel_mat(y, x, gamma)) / x.shape[0] / y.shape[0]
    mmd = kxx + kyy - 2 * kxy
    return mmd


