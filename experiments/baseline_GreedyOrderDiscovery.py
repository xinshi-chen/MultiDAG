import numpy as np
import os, sys, time
import pickle
from multidag.dag_utils import is_dag
from sklearn import linear_model
from multidag.common.cmd_args import cmd_args
from tqdm import tqdm


class GreedyOrderDicovery(object):
    def __init__(self, X):
        assert len(X.shape) == 3
        assert X.shape[0] == 1 # we only use it to solve K = 1
        self.X = X.squeeze(0)
        self.n = X.shape[1]
        self.p = X.shape[2]
        self.A = None

    def train(self):
        self.A = np.zeros((self.p, self.p))
        order = self._get_order()
        self._lasso(order)
        return self.A

    def _get_order(self):
        X = self.X.copy()
        mask_idx = np.zeros(self.p)
        ordered_idx = []
        for p in range(self.p - 1):
            var = np.var(X, axis=0)
            idx = np.argmin(var + mask_idx * 1e8)
            beta = np.sum(X[:, idx:idx+1] * X, axis=0) / (np.sum(X[:, idx]**2))
            X -= X[:, idx:idx+1] * beta.reshape(1, self.p)
            mask_idx[idx] += 1
            ordered_idx.append(idx)
        ordered_idx.append(np.argmin(mask_idx))
        return ordered_idx

    def _lasso(self, order):
        alpha = np.sqrt(8 * np.log(self.p) / self.n)
        clf = linear_model.Lasso(alpha=alpha, max_iter=100000)
        for j in range(1, self.p):
            X = self.X[:, order[:j]]
            Y = self.X[:, j]
            clf.fit(X, Y)
            self.A[order[:j], j] = clf.coef_


if __name__ == '__main__':
    p = cmd_args.p
    n = cmd_args.n_sample
    K = cmd_args.K
    s0 = cmd_args.s0
    s = cmd_args.s
    d = cmd_args.d
    group_size = cmd_args.group_size
    group_start = cmd_args.group_start
    group_end = cmd_args.group_end
    w_range = (0.5, 2.0)
    hp_dict = {'p': p,
               'n': n,
               'K': K,
               's': s,
               's0': s0,
               'd': d,
               'w_range_l': w_range[0],
               'w_range_u': w_range[1]}

    hp = ''
    for key in hp_dict:
        hp += key + '-' + str(hp_dict[key]) + '_'
    hp = hp[:-1]
    data_dir = os.path.join('../data', hp)
    G = pickle.load(open(data_dir + '/DAGs.pkl', 'rb'))
    X = pickle.load(open(data_dir + '/samples_gid.pkl', 'rb')).numpy()


    A = np.zeros((group_end - group_start, p, p))
    T = []
    progress_bar = tqdm(range(int((group_end - group_start) / group_size)))
    nnz_A = []
    for i in progress_bar:
        myGOD = GreedyOrderDicovery(X[group_start + i * group_size: group_start + (i+1) * group_size])
        t0 = time.time()
        A[i * group_size: (i+1) * group_size] = myGOD.train()
        t1 = time.time()
        T.append(t1 - t0)
        nnz_A.append((np.abs(myGOD.A) > 0).sum())
        progress_bar.set_description(f'[nnz_A: {np.mean(nnz_A):.2f}]')

    save_dir = os.path.join('saved_models', hp, 'GOD')
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, f'{group_size}_{group_start}-{group_end}.pkl'), 'wb') as handle:
        pickle.dump([T, A], handle)
