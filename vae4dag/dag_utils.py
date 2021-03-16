import scipy.optimize as sopt
import scipy.linalg as slin
import numpy as np
import networkx as nx
from vae4dag.common.consts import DEVICE
import torch
from notears.linear import notears_linear
from tqdm import tqdm
import math


def run_notears_linear(X):
    """
    :param X: [m, n, d]
    :param d:
    :return: W_est: [m, d, d]
    """
    assert len(X.shape) == 3
    num_dag = X.shape[0]
    d = X.shape[2]
    W_est = np.zeros([num_dag, d, d])
    progress_bar = tqdm(range(num_dag))
    for i in progress_bar:
        W_est[i] = notears_linear(X[i], lambda1=0.1, loss_type='l2')
        assert is_dag(W_est[i])
    return W_est.astype(np.float32)


def project_to_dag(W, sparsity=1.0, max_iter=20, h_tol=1e-3, rho_max=1e+16, w_threshold=0.1):
    """
    :param W: (np.ndarray) [d, d] matrix as a general directed graph, not necessarily acyclic
    :return:
        W: (np.ndarray) [d, d] approximate projection to DAGs
        return None if it takes to long to project to DAGs
    """
    for _ in range(5):  # run at most 5 times

        try:
            W, P = project_notears(W, sparsity, max_iter, h_tol, rho_max, w_threshold)
        except ValueError:
            # in case of some numerical instability error
            return None, None

        if is_dag(P):
            return W, P

    return None, None


def is_dag(W: np.ndarray):
    if W is not None:
        G = nx.DiGraph(W)
        return nx.is_directed_acyclic_graph(G)
    else:
        return False


def project_notears(X, sparsity=1.0, max_iter=100, h_tol=1e-3, rho_max=1e+16, w_threshold=0.1):
    """
    Projection to the space of DAGs. Solved based on the 'DASs with NO TEARS' paper.
    Implemented based on https://github.com/xunzheng/notears/blob/master/notears/linear.py
    Solve min_W L(W;X) + lambda1 ‖W‖_1 s.t. h(W) = 0 using augmented Lagrangian.

    To perform projection, the loss is L(W; X) = 1/2 ‖W - X‖_F^2.
    When lambda1 > 0, then this is not a pure projection, but with l1 regularization.

    Args:
        X (np.ndarray): [d, d] matrix as a general directed graph, not necessarily acyclic
        sparsity (float): l1 penalty parameter
        max_iter (int): max num of dual ascent steps
        h_tol (float): exit if |h(w_est)| <= htol
        rho_max (float): exit if rho >= rho_max
        w_threshold (float): drop edge if |weight| < threshold
    Returns:
        W_est (np.ndarray): [d, d] approximate projection to DAGs
    """

    n, d = X.shape
    assert n == d

    def _loss(W):
        """Evaluate value and gradient of loss."""
        R = W - X
        loss = 0.5 * (R ** 2).sum()
        G_loss = R

        return loss, G_loss

    def _h(W):
        """Evaluate value and gradient of acyclicity constraint."""
        E = slin.expm(W * W)  # (sZheng et al. 2018)
        h = np.trace(E) - d
        #     # A different formulation, slightly faster at the cost of numerical stability
        #     M = np.eye(d) + W * W / d  # (Yu et al. 2019)
        #     E = np.linalg.matrix_power(M, d - 1)
        #     h = (E.T * M).sum() - d
        G_h = E.T * W * 2
        return h, G_h

    def _adj(w):
        """Convert doubled variables ([2 d^2] array) back to original variables ([d, d] matrix)."""
        return (w[:d * d] - w[d * d:]).reshape([d, d])

    def _func(w):
        """Evaluate value and gradient of augmented Lagrangian for doubled variables ([2 d^2] array)."""
        W = _adj(w)
        loss, G_loss = _loss(W)
        h, G_h = _h(W)
        obj = loss + 0.5 * rho * h * h + alpha * h + sparsity * w.sum()
        G_smooth = G_loss + (rho * h + alpha) * G_h
        g_obj = np.concatenate((G_smooth + sparsity, - G_smooth + sparsity), axis=None)
        return obj, g_obj

    w_est, rho, alpha, h = np.ones(2 * d * d), 1.0, 0.0, np.inf  # double w_est into (w_pos, w_neg)
    bnds = [(0, 0) if i == j else (0, None) for _ in range(2) for i in range(d) for j in range(d)]

    for _ in range(max_iter):
        w_new, h_new = None, None
        while rho < rho_max:
            sol = sopt.minimize(_func, w_est, method='L-BFGS-B', jac=True, bounds=bnds)
            w_new = sol.x
            h_new, _ = _h(_adj(w_new))
            if h_new > 0.25 * h:
                rho *= 10
            else:
                break
        w_est, h = w_new, h_new
        alpha += rho * h
        if h <= h_tol or rho >= rho_max:
            break
    W_est = _adj(w_est)
    P = np.abs(W_est) >= w_threshold
    W_est.fill(0)
    W_est[P] = X[P]
    return W_est, P


def sampler(W, n, f, g=None):
    """
    sample n samples from the probablistic model defined by W, f, and g

    :param W: weighted adjacency matrix. size=[d, d]
    :param n: number of samples

    :return: X: [n, d] sample matrix
    """

    if isinstance(W, np.ndarray):
        W = torch.tensor(W)
    elif not torch.is_tensor(W):
        raise NotImplementedError('Adjacency matrix should be np.ndarray or torch.tensor.')

    d = W.shape[0]

    X = torch.zeros([n, d])
    neg_log_likelihood = torch.zeros([n, d])

    z = torch.normal(0, 1, size=(n, d)).float()

    # get the topological order of the DAG
    G = nx.DiGraph(W.detach().cpu().numpy())
    ordered_vertices = list(nx.topological_sort(G))
    assert len(ordered_vertices) == d
    for j in ordered_vertices:

        WX = W[:, j] * X  # [n, d]
        m_j = f[j](WX).view(n)  # mean [n]

        if g is not None:
            sigma_j = torch.abs(g[j](WX).view(n))
            log_z = 0.5 * math.log(2 * math.pi) + torch.log(sigma_j)
        else:
            sigma_j = 1.0
            log_z = 0.5 * math.log(2 * math.pi)

        X[:, j] = m_j + z[:, j] * sigma_j
        neg_log_likelihood[:, j] = log_z + 0.5 * ((X[:, j] - m_j) / sigma_j) ** 2

    return X, torch.sum(neg_log_likelihood, dim=-1).mean().item()


if __name__ == '__main__':
    d = 2
    threshold = 0.1
    sparsity = 1.0
    x = np.random.uniform(low=-2, high=2, size=[d, d])
    x[np.abs(x)<threshold] = 0
    print(x)
    y, p = project_to_dag(x, max_iter=10, w_threshold=threshold, sparsity=sparsity)
    print('projected')
    print(y)
    print(p)
    print('dagness')
    print(is_dag(p))

    n = 30
    mu = np.zeros(d)
    sigma = np.ones(d)
    x_np = sampler(y, n, mu, sigma)
    print(x_np)
    x_torch = sampler(torch.tensor(y), n, torch.tensor(mu), torch.tensor(sigma))
    print(x_torch)

    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.scatter(x_np[:,0], x_np[:,1])
    plt.scatter(x_torch.detach().numpy()[:,0], x_torch.detach().numpy()[:,1])
    plt.show()