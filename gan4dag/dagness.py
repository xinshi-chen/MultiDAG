import scipy.optimize as sopt
import scipy.linalg as slin
import numpy as np
import networkx as nx


def project_to_dag(W, sparsity=1.0, max_iter=10, h_tol=1e-3, rho_max=1e+16, w_threshold=0.1):
    """
    :param W: (np.ndarray) [d, d] matrix as a general directed graph, not necessarily acyclic
    :return:
        W: (np.ndarray) [d, d] approximate projection to DAGs
        return None if it takes to long to project to DAGs
    """
    for _ in range(5):  # run at most 5 times
        W, P = project_notears(W, sparsity, max_iter, h_tol, rho_max, w_threshold)
        if is_dag(P):
            return W, P
    # raise Exception("Run too long for DAG projection.")
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


if __name__ == '__main__':
    d = 10
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
