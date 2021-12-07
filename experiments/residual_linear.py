import torch
import os, pickle
from multidag.common.cmd_args import cmd_args
from multidag.sergio_dataset import SergioDataset
import random
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso


def causal_order(task_expr):
    """
    Returns a causal ordering of the nodes as a Numpy array where 0 -> 1 -> ...
    """
    d = task_expr.shape[1]
    order = []
    unordered = list(range(d))
    var_all = np.var(task_expr, axis=0)
    top_i = np.argsort(var_all)[0]  # Min residual variance
    order.append(top_i)
    del unordered[top_i]
    while len(order) < d:
        X = task_expr[:, order]
        y = task_expr[:, unordered]
        reg = LinearRegression().fit(X, y)
        residuals = y - reg.predict(X)
        var_residuals = np.var(residuals, axis=0)
        top_i = np.argsort(var_residuals)[0]  # Min residual variance
        order.append(unordered[top_i])
        del unordered[top_i]
    return np.array(order)


def graphical_lasso(task_expr, order, lam=0.1):
    """
    Returns an adjacency matrix of the DAG discovered by progressing through the
    causal ordering and doing graphical lasso at each level
    """
    d = task_expr.shape[1]
    adj = np.zeros((d, d))
    for i in range(1, d):
        parents = order[0:i]
        child = order[i]
        X = task_expr[:, parents]
        y = task_expr[:, child]
        reg = Lasso(alpha=lam).fit(X, y)
        coef = np.round(reg.coef_, 4)
        adj[parents, child] = coef
    return adj


def solve_residual(cmd_args, db, group_size=1, group_start=0):
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    # G_DAG
    assert group_size <= db.K
    hyperparameter = {'rho': cmd_args.rho, 'lambda': cmd_args.ld, 'c': cmd_args.c, 'gamma': cmd_args.gamma,
                      'eta': cmd_args.eta, 'mu': cmd_args.mu, 'dual_interval': cmd_args.dual_interval,
                      'alpha': cmd_args.alpha}
    hp = ''
    for key in hyperparameter:
        hp += key + '-' + f'{hyperparameter[key]}' + '_'
    hp = hp[:-1]

    G = []
    for X in db.load_by_task():
        order = causal_order(X)
        adj = graphical_lasso(X, order, lam=cmd_args.ld)
        G.append(adj)
    G = np.array(G)
    T = np.ones(G.shape)
    models = [{'G': G, 'T': T}]

    model_save_root = './saved_baselines/' + db.hp + '/' + hp
    if not os.path.isdir(model_save_root):
        os.makedirs(model_save_root)
    name = f'multidag_group_size-{group_size}-{group_start}-{group_start + group_size}.pkl'
    model_save_dir = model_save_root + '/' + name
    with open(model_save_dir, 'wb') as handle:
        pickle.dump(models, handle)



if __name__ == '__main__':
    db = SergioDataset(cmd_args.real_dir)
    print(f'*** solving {db.hp}_group_size-{cmd_args.group_size}-{cmd_args.group_start}-{cmd_args.group_start+cmd_args.group_size} ***')
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    X = db.X[cmd_args.group_start:cmd_args.group_start + cmd_args.group_size].detach().numpy()
    G = db.G[cmd_args.group_start:cmd_args.group_start + cmd_args.group_size]
    real_se = np.square(X - X@G).sum(axis=-1).mean()
    real_gn = np.linalg.norm(G, axis=0).sum()
    print('real se: ', real_se)
    print('real group norm: ', real_gn)
    solve_residual(cmd_args, db, group_size=cmd_args.group_size, group_start=cmd_args.group_start)
