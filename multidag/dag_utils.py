import scipy.optimize as sopt
import scipy.linalg as slin
import numpy as np
import networkx as nx
from multidag.common.consts import DEVICE
import torch
# from notears.linear import notears_linear
from tqdm import tqdm
import math
import torch.nn.functional as F


def matrix_poly(W):
    if len(W.shape) == 2:
        d = W.shape[0]
        assert d == W.shape[1]
        x = torch.eye(d).to(DEVICE) + 1/d * W
        return torch.matrix_power(x, d)
    elif len(W.shape) == 3:
        m, d = W.shape[0], W.shape[1]
        x = torch.eye(d).unsqueeze(0).repeat(m, 1, 1).detach().to(DEVICE) + 1/d * W
        return torch.matrix_power(x, d)
    else:
        raise NotImplementedError('Shape should has length 2 or 3.')


def DAGGNN_h_W(W):
    expd_W = matrix_poly(W * W)
    if len(W.shape) == 2:
        h_W = torch.trace(expd_W) - d
    elif len(W.shape) == 3:
        h_W = torch.einsum('bii->b', expd_W) - d
    else:
        raise NotImplementedError('Shape should has length 2 or 3.')
    return h_W


class NOTEARS_h_W(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):

        """
        input: [batch, d, d] tensor containing batch many matrices
        """

        if len(input.shape) == 2:
            d = input.shape[0]
            e_W_W = torch.matrix_exp(input * input)
            tr_e_W_W = torch.trace(e_W_W)
        elif len(input.shape) == 3:
            d = input.shape[1]
            assert d == input.shape[2]
            e_W_W = torch.matrix_exp(input * input)
            tr_e_W_W = torch.einsum('bii->b', e_W_W)  # [batch]
        else:
            raise NotImplementedError('Shape should has length 2 or 3.')
        ctx.save_for_backward(input, e_W_W)

        return tr_e_W_W - d

    @staticmethod
    def backward(ctx, grad_output):

        input, e_W_W = ctx.saved_tensors
        if len(input.shape) == 2:
            grad_input = e_W_W.t() * 2 * input
            return grad_input * grad_output
        elif len(input.shape) == 3:
            m = input.shape[0]
            grad_input = e_W_W.permute(0, 2, 1) * 2 * input  # [batch, d, d]
            return grad_input * grad_output.view(m, 1, 1)


h_W = {'notears': NOTEARS_h_W.apply,
       'daggnn': DAGGNN_h_W}


class TraceExpm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        """
        input : A = [d, d] tensor
        output: tr(e^A)
        """
        E_A = torch.matrix_exp(input)
        tr_E_A = torch.trace(E_A)
        ctx.save_for_backward(E_A)
        return tr_E_A

    @staticmethod
    def backward(ctx, grad_output):
        E_A, = ctx.saved_tensors
        grad_input = grad_output * E_A.t()
        return grad_input


trace_expm = TraceExpm.apply


# def run_notears_linear(X):
#     """
#     :param X: [m, n, d]
#     :param d:
#     :return: W_est: [m, d, d]
#     """
#     assert len(X.shape) == 3
#     num_dag = X.shape[0]
#     d = X.shape[2]
#     W_est = np.zeros([num_dag, d, d])
#     progress_bar = tqdm(range(num_dag))
#     for i in progress_bar:
#         W_est[i] = notears_linear(X[i], lambda1=0.1, loss_type='l2')
#         assert is_dag(W_est[i])
#     return W_est.astype(np.float32)


def project_to_dag(W, sparsity=1.0, max_iter=20, h_tol=1e-3, rho_max=1e+16, w_threshold=0.1):
    """
    :param W: (np.ndarray) [d, d] matrix as a general directed graph, not necessarily acyclic
    :return:
        W: (np.ndarray) [d, d] approximate projection to DAGs
        return None if it takes to long to project to DAGs
    """
    W = W * (np.abs(W) > w_threshold)
    for _ in range(5):  # run at most 5 times

        try:
            W, P = project_notears(W, sparsity, max_iter, h_tol, rho_max, w_threshold)
        except ValueError:
            print('numerical instability error')
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

        # use torch.matrix_exp

        # E = slin.expm(W * W)  # (sZheng et al. 2018)
        E = torch.matrix_exp(torch.tensor(W * W)).numpy()
        h = np.trace(E) - d

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


def sampler(W, n, f=None, g=None):
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
        if f is not None:
            m_j = f[j](WX).view(n)
        else:
            m_j = WX.sum(dim=-1)   # linear model

        if g is not None:
            sigma_j = torch.abs(g[j](WX).view(n))
            log_z = 0.5 * math.log(2 * math.pi) + torch.log(sigma_j)
        else:
            sigma_j = 1.0
            log_z = 0.5 * math.log(2 * math.pi)

        X[:, j] = m_j + z[:, j] * sigma_j
        neg_log_likelihood[:, j] = log_z + 0.5 * ((X[:, j] - m_j) / sigma_j) ** 2

    return X, torch.sum(neg_log_likelihood, dim=-1)


def count_accuracy(B_true, B_est):
    """Compute various accuracy metrics for B_est.
        true positive = predicted association exists in condition in correct direction
        reverse = predicted association exists in condition in opposite direction
        false positive = predicted association does not exist in condition
        Args:
            B_true (np.ndarray): [d, d] ground truth graph, {0, 1}
            B_est (np.ndarray): [d, d] estimate, {0, 1}
        Returns:
            fdr: (reverse + false positive) / prediction positive
            tpr: (true positive) / condition positive
            fpr: (reverse + false positive) / condition negative
            shd: undirected extra + undirected missing + reverse
            nnz: prediction positive
        """
    if (B_est == -1).any():  # cpdag
        if not ((B_est == 0) | (B_est == 1) | (B_est == -1)).all():
            raise ValueError('B_est should take value in {0,1,-1}')
        if ((B_est == -1) & (B_est.T == -1)).any():
            raise ValueError('undirected edge should only appear once')
    else:  # dag
        if not ((B_est == 0) | (B_est == 1)).all():
            raise ValueError('B_est should take value in {0,1}')
        if not is_dag(B_est):
            raise ValueError('B_est should be a DAG')
    d = B_true.shape[0]
    # linear index of nonzeros
    pred_und = np.flatnonzero(B_est == -1)
    pred = np.flatnonzero(B_est == 1)
    cond = np.flatnonzero(B_true)
    cond_reversed = np.flatnonzero(B_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # treat undirected edge favorably
    true_pos_und = np.intersect1d(pred_und, cond_skeleton, assume_unique=True)
    true_pos = np.concatenate([true_pos, true_pos_und])
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    false_pos_und = np.setdiff1d(pred_und, cond_skeleton, assume_unique=True)
    false_pos = np.concatenate([false_pos, false_pos_und])
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred) + len(pred_und)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    fdr = float(len(reverse) + len(false_pos)) / max(pred_size, 1)
    tpr = float(len(true_pos)) / max(len(cond), 1)
    fpr = float(len(reverse) + len(false_pos)) / max(cond_neg_size, 1)
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_est + B_est.T))
    cond_lower = np.flatnonzero(np.tril(B_true + B_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size}


if __name__ == '__main__':
    d = 2
    threshold = 0.1
    sparsity = 1.0
    x = np.random.uniform(low=-2, high=2, size=[d, d])
    x[np.abs(x)<threshold] = 0
    print(x)
    # y, p = project_to_dag(x, max_iter=10, w_threshold=threshold, sparsity=sparsity)
    # print('projected')
    # print(y)
    # print(p)
    # print('dagness')
    # print(is_dag(p))
    #
    # n = 30
    # mu = np.zeros(d)
    # sigma = np.ones(d)
    # x_np = sampler(y, n, mu, sigma)
    # print(x_np)
    # x_torch = sampler(torch.tensor(y), n, torch.tensor(mu), torch.tensor(sigma))
    # print(x_torch)
    #
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # plt.scatter(x_np[:,0], x_np[:,1])
    # plt.scatter(x_torch.detach().numpy()[:,0], x_torch.detach().numpy()[:,1])
    # plt.show()
