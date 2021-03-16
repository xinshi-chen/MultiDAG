import torch
from vae4dag.common.consts import DEVICE, OPTIMIZER
from vae4dag.trainer import Trainer
from vae4dag.eval import Eval
from vae4dag.common.cmd_args import cmd_args
from vae4dag.data_generator import Dataset
from vae4dag.dag_utils import is_dag
import random
import numpy as np
from vae4dag.model import Encoder, Decoder
import torch.nn as nn
from notears.nonlinear import NotearsMLP, LBFGSBScipy, squared_loss
from tqdm import tqdm
import math


# redefine this function to return the model
def notears_nonlinear(model: nn.Module,
                      X,
                      lambda1: float = 0.,
                      lambda2: float = 0.,
                      max_iter: int = 100,
                      h_tol: float = 1e-8,
                      rho_max: float = 1e+16,
                      w_threshold: float = 0.3):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, lambda1, lambda2,
                                         rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est


def dual_ascent_step(model, X_torch, lambda1, lambda2, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch)
            loss = squared_loss(X_hat, X_torch)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l2_reg + l1_reg
            primal_obj.backward()
            return primal_obj
        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def notears_mlp(X, X_test):
    """
    X: [num_dag, n, d]
    """
    np.set_printoptions(precision=3)
    num_dag = X.shape[0]
    d = X.shape[2]
    W_est = np.zeros([num_dag, d, d])
    nll_est = np.zeros([num_dag])
    progress_bar = tqdm(range(num_dag))
    for i in progress_bar:
        model = NotearsMLP(dims=[d, 32, 16, 1], bias=True)
        W_est[i] = notears_nonlinear(model, X[i], lambda1=0.01, lambda2=0.01)
        assert is_dag(W_est[i])

        # compute likelihood

        with torch.no_grad():
            X_test_i = X_test[i]
            X_mean = model(X_test_i)
            log_z = 0.5 * math.log(2 * math.pi)
            neg_log_likelihood = log_z + 0.5 * (X_test_i - X_mean) ** 2  # [n, d]
            neg_log_likelihood = torch.sum(neg_log_likelihood, dim=-1)

        nll_est[i] = neg_log_likelihood.mean().item()

        progress_bar.set_description('%d: [nll: %.3f]' % (i, nll_est[i]))

    return W_est, nll_est


if __name__ == '__main__':

    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    # ---------------------
    #  Hyperparameters
    # ---------------------

    num_dag = cmd_args.num_dag
    num_sample = cmd_args.num_sample
    num_sample_gen = cmd_args.num_sample_gen
    threshold = cmd_args.threshold
    sparsity = cmd_args.sparsity
    d = cmd_args.d

    # ---------------------
    #  Synthetic Dataset
    # ---------------------

    print('*** Loading data ***')
    db = Dataset(d, W_sparsity=sparsity, W_threshold=threshold, num_dags=num_dag, num_sample=num_sample,
                 f_hidden_dims=cmd_args.true_f_hidden_dim, f_act=cmd_args.true_f_act, g_hidden_dims=None, g_act=None,
                 verbose=True)

    # ---------------------
    #  Run On Training Data
    # ---------------------

    X, nll = db.train_data['data']
    W_est, nll_est = notears_mlp(X, X_test=X)
    print('NLL true: %.3f, est: %.3f', (nll.mean().item(), nll_est.mean()))




