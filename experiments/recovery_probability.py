import numpy as np
import pickle
import matplotlib.pyplot as plt
from multidag.data_generator import Dataset
from multidag.common.cmd_args import cmd_args
from multidag.dag_utils import is_dag, project_to_dag
from multidag.model import G_DAG

def success(B_est, perm):
    '''
    check whether the topo order is successfully recovered
    '''
    if not is_dag(B_est):
        print('B_est is not DAG, use projection')
        B_est, _ = project_to_dag(B_est)
    inv = np.linalg.inv(perm)
    target = inv.T.dot(B_est).dot(inv)
    return np.abs(np.triu(target, 1)).sum() < 1e-6

def accuracy(result):
    accs = []
    k = result.shape[0]
    for i in range(1-k, k):
        accs.append(np.tril(np.triu(np.flip(result, axis=0), i), i).sum() / (k - np.abs(i)))
    return accs

if __name__ == '__main__':
    results = {20:[], 40: [], 80: [], 160: [], 320: [], 640: []}
    for n in results:
        db = Dataset(p=cmd_args.p,
                     n=n,
                     K=cmd_args.K,
                     s0=cmd_args.s0,
                     s=cmd_args.s,
                     d=cmd_args.d,
                     w_range=(0.5, 2.0), verbose=True)
        perm = db.Perm
        group_size = 1
        while group_size <= cmd_args.K:
            g_dag = G_DAG(num_dags=group_size, p=cmd_args.p)
            recovery = []
            for i in range(db.K // group_size):
                with open(
                        f'./saved_models/p-{cmd_args.p}_n-{n}_K-{cmd_args.K}_s-{cmd_args.s}_s0-{cmd_args.s0}_'
                        f'd-{cmd_args.d}_w_range_l-0.5_w_range_u-2.0/rho-{cmd_args.rho}_lambda-{cmd_args.ld}_'
                        f'c-{cmd_args.c}_gamma-{cmd_args.gamma}_'
                        f'eta-{cmd_args.eta}_mu-{cmd_args.mu}_dual_interval-{cmd_args.dual_interval}/multidag_'
                        f'group_size-{group_size}-{i * group_size}-{(i + 1) * group_size}.pkl', 'rb') as handle:
                    model = pickle.load(handle)[0]
                B_est = np.abs(model['G'] * model['T'])
                B_est[B_est < cmd_args.threshold] = 0
                B_est = np.sign(B_est)
                for k in range(group_size):
                    assert is_dag(B_est[k])
                    recovery.append(success(B_est[k], perm))
            results[n].append(np.mean(recovery))
            group_size *= 2

    result = np.array([results[key] for key in results])
    print(result)
    y = accuracy(result)
    x = [2**(i/2) for i in range(len(y))]
    plt.plot(x, y)
    plt.show()
