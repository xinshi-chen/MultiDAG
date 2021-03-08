import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from gan4dag.common.consts import DEVICE


# -----------------
#  Single pair MMD
# -----------------

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
        median_dist2 = torch.zeros(size=[m]).to(DEVICE)
        for i in range(m):
            d = dist2[i][dist2[i] > 0]
            median_dist2[i] = torch.median(d)
        gamma = 0.5 / median_dist2 / bandwidth
        return gamma


def get_kernel_mat(x, landmarks, gamma):
    """
    Reference: https://github.com/xinshi-chen/ParticleFlowBayesRule/blob/master/pfbayes/common/distributions.py#L68
    """
    d = pairwise_distances(x, landmarks)
    k = torch.exp(d * -gamma)
    k = k.view(x.shape[0], -1)
    return k


# -----------------
#  Batch MMD
# -----------------

def MMD_batch(x, y, bandwidth=1.0):
    # y [m, n, d]
    # x [m2, n2, d]

    # return mmd [m2, m]

    m2, n2 = x.shape[0], x.shape[1]
    m, n = y.shape[0], y.shape[1]

    y = y.detach()

    gamma = get_gamma_batch(y, bandwidth).detach()  # [m]
    kxx = kxx_batch(x, gamma, landmark=False)   # [m2, m, n2, n2]
    kxx = torch.sum(kxx.view(m2, m, n2 * n2), dim=-1) / n2 / (n2 - 1)  # [m2, m]

    kyy = kxx_batch(y, gamma, landmark=True)   # [m, n, n]
    kyy = (torch.sum(kyy.view(m, -1), dim=-1) / n / (n - 1)).detach()  # [m]

    kxy = kxy_batch(x, y, gamma)  # [m, m2, n, n2]
    kxy = torch.sum(kxy.reshape(m, m2, n*n2), dim=-1).transpose(0, 1) / n / n2  # [m2, m]

    mmd = kxx + kyy.view(1, m).detach() - 2.0 * kxy
    return mmd


def kxx_batch(x, gamma, landmark=False):
    """
    x [m2, n, d]
    gamma [m]
    landmark: whether x data is viewed as landmark

    if landmark:
    :return [m2, n, n]
    else:
    :return [m2, m, n, n]
    """

    m2 = x.shape[0]
    n = x.shape[1]
    m = len(gamma)
    if landmark:
        assert m2 == m

    d = pairwise_distances_batch(x)  # [m2, n, n]
    if landmark:
        d = torch.einsum('mjk,m->mjk', d, -gamma)  # [m, n, n]
    else:
        d = torch.einsum('ijk,m->imjk', d, -gamma)  # [m2, m, n, n]
    k = torch.exp(d)

    # make diagonal zero
    idx = torch.tensor(range(n)).to(DEVICE)
    if landmark:
        k[:, idx, idx] = 0
    else:
        k[:, :, idx, idx] = 0

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


if __name__ == '__main__':
    m = 8
    n = 3
    m2 = 7
    n2 = 6
    d = 5

    x_fake = torch.rand([m2, n2, d]).to(DEVICE)
    x_real = torch.rand([m, n, d]).to(DEVICE)

    mmd_seq = torch.zeros(size=[m2, m])
    for i in range(m2):
        for j in range(m):
            mmd_seq[i, j] = MMD(x_fake[i], x_real[j])
    print(mmd_seq)

    mmd_batch = MMD_batch(x_fake, x_real)
    print(mmd_batch)
