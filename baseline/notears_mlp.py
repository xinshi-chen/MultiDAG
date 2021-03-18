import torch
from vae4dag.common.cmd_args import cmd_args
from vae4dag.data_generator import GenDataset
from vae4dag.dag_utils import is_dag
from vae4dag.common.utils import weights_init
import random
import numpy as np
import torch.nn as nn
from notears.nonlinear import NotearsMLP, LBFGSBScipy, squared_loss
from tqdm import tqdm
import math
import pickle as pkl


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


def notears_mlp(X, X_test, model_dump=None):
    """
    X: [num_dag, n, d]
    """
    np.set_printoptions(precision=3)
    num_dag = X.shape[0]
    d = X.shape[2]
    W_est = np.zeros([num_dag, d, d])
    nll_eval = np.zeros([num_dag])
    nll_train = np.zeros([num_dag])
    progress_bar = tqdm(range(num_dag))

    hidden_dims = [d, 32, 16, 1]

    for i in progress_bar:
        model = NotearsMLP(dims=hidden_dims, bias=True)
        weights_init(model)
        W_est[i] = notears_nonlinear(model, X[i], lambda1=0.01, lambda2=0.01)
        assert is_dag(W_est[i])

        # compute likelihood

        def get_nll(xx):
            with torch.no_grad():
                x_mean = model(xx)
                log_z = 0.5 * math.log(2 * math.pi)
                neg_log_likelihood = log_z + 0.5 * (xx - x_mean) ** 2  # [n, d]
                neg_log_likelihood = torch.sum(neg_log_likelihood, dim=-1).mean().item()
            return neg_log_likelihood

        nll_eval[i] = get_nll(X_test[i])
        nll_train[i] = get_nll(X[i])

        progress_bar.set_description('%d: [nll train: %.3f] [nll test: %.3f]' % (i, nll_train[i], nll_eval[i]))

        if model_dump is not None:
            dump = model_dump + "-".join(map(str, hidden_dims)) + '-%d.dump' % i
            torch.save(model.state_dict(), dump)

    return W_est, nll_train, nll_eval


if __name__ == '__main__':

    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    # ---------------------
    #  Hyperparameters
    # ---------------------

    num_sample = cmd_args.num_sample
    threshold = cmd_args.threshold
    sparsity = cmd_args.sparsity
    d = cmd_args.d

    # ---------------------
    #  Synthetic Dataset
    # ---------------------

    print('*** Loading data ***')
    db = GenDataset(d, num_sample, W_sparsity=sparsity, W_threshold=threshold, f_hidden_dims=cmd_args.true_f_hidden_dim,
                    f_act=cmd_args.true_f_act, g_hidden_dims=None, g_act=None, verbose=True, num_test=cmd_args.num_sample_test,
                    num_dags={'train': cmd_args.num_train,
                              'vali': cmd_args.num_vali,
                              'test': cmd_args.num_test})

    # ---------------------
    #  Run On Test Data
    # ---------------------
    X, nll = db.static_data['test']
    k = math.floor(num_sample * cmd_args.p)
    num_eval = cmd_args.num_sample_test - k

    X_in, true_nll_in = X[:, :k, :], nll[:, :k]
    X_eval, true_nll_eval = X[:, k:, :], nll[:, k:]

    model_dump = cmd_args.save_dir + '/notears_mlp-' + db.dataset_hp
    W_est, nll_train, nll_test = notears_mlp(X=X_in, X_test=X_eval, model_dump=model_dump)
    print(true_nll_eval)
    print(nll_test)

    print('*** On observed samples ***')
    print('NLL true: %.3f, estimated: %.3f' % (true_nll_in.mean(), nll_train.mean()))
    print('*** On test samples ***')
    print('NLL true: %.3f, estimated: %.3f' % (true_nll_eval.mean(), nll_test.mean()))

    filename = 'results-notears_mlp-' + db.dataset_hp + '.pkl'
    with open(filename, 'wb') as f:
        pkl.dump([W_est, nll_train, nll_test], f)


