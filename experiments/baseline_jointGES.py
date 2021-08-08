import numpy as np
import os, sys, time
import pickle
from multidag.dag_utils import is_dag
from sklearn import linear_model
from multidag.common.cmd_args import cmd_args
from tqdm import tqdm


class jointGES(object):
    def __init__(self, X, d=5):
        assert len(X.shape) == 3
        self.X = X
        self.K = X.shape[0]
        self.n = X.shape[1]
        self.p = X.shape[2]
        self.d = d
        self.lamda = None
        self.A = None
        self.G = None
        self.G_temp = None

    def train(self, alpha=None, lamda=None):
        self.lamda = 3 * np.log(self.p) / self.n if lamda is None else lamda
        self._GES()
        self._lasso(alpha)
        return self.A

    def _GES(self):
        '''
        When count == 0, flip the phase
        When count == 0 two times, stop greedy search
        '''
        self.G = np.zeros((self.p, self.p))
        phase = 1
        flag, count = 1, 1
        while(flag):
            if count == 0:
                flag = 0
                phase *= -1
            count = 0
            for i in range(self.p):
                for j in range(self.p):
                    dE = self._deltaE(i, j, phase)
                    if dE < 0:
                        self._updateG(i, j, phase)
            if count:
                flag = 1

    def _lasso(self, alpha=None):
        alpha = 2 * np.sqrt(np.log(self.p) / self.n) if alpha is None else alpha
        self.A = np.zeros((self.K, self.p, self.p))
        clf = linear_model.Lasso(alpha=alpha, max_iter=10000)
        for k in range(self.K):
            for j in range(self.p):
                if self.G[:, j].sum() == 0:
                    continue
                X = self.X[k][:, self.G[:, j].astype(bool)]
                Y = self.X[k, :, j]
                clf.fit(X, Y)
                self.A[k, self.G[:, j].astype(bool), j] = clf.coef_


    def _deltaE(self, i=0, j=0, phase=1):
        # phase: 1 means adding, -1 means deleting
        if not self._is_valid(i, j, phase):
            return 0
        dE = phase * self.lamda
        for k in range(self.K):
            if self.G[:, j].sum() > 0:
                X = self.X[k][:, self.G[:, j].astype(bool)]
                X_temp = self.X[k][:, self.G_temp[:, j].astype(bool)]
                Y = self.X[k, :, j:j+1]
                dE += (np.log(np.sum(np.square(Y - X_temp @ np.linalg.inv(X_temp.T @ X_temp) @ X_temp.T @ Y))) -
                      np.log(np.sum(np.square(Y - X @ np.linalg.inv(X.T @ X) @ X.T @ Y)))) / self.K
            else:
                X_temp = self.X[k][:, self.G_temp[:, j].astype(bool)]
                Y = self.X[k, :, j:j+1]
                dE += (np.log(np.sum(np.square(Y - X_temp @ np.linalg.inv(X_temp.T @ X_temp) @ X_temp.T @ Y))) -
                       np.log(np.sum(np.square(Y))))/ self.K
        return dE

    def _updateG(self, i=0, j=0, phase=1):
        if phase == 1:
            self.G[i, j] = 1
        else:
            self.G[i, j] = 0

    def _is_valid(self, i=0, j=0, phase=1):
        if phase == -1:
            if self.G[i, j] == 1:
                self.G_temp = self.G.copy()
                self.G_temp[i,j] = 0
                return 1
            else:
                return 0
        elif phase == 1:
            if self.G[i, j] == 1:
                return 0
            else:
                self.G_temp = self.G.copy()
                self.G_temp[i, j] = 1
                if self.G_temp[:, j].sum() > self.d:
                    return 0
                return is_dag(self.G_temp)

if __name__ == '__main__':
    p = cmd_args.p
    n = cmd_args.n_sample
    K = cmd_args.K
    s0 = cmd_args.s0
    s = cmd_args.s
    d = cmd_args.d
    group_size = cmd_args.group_size
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

    t0 = time.time()
    A = np.zeros((K, p, p))
    for i in tqdm(range(int(K / group_size))):
        ges = jointGES(X[group_size*(i-1): group_size*i], d=d)
        A[group_size*(i-1): group_size*i] = ges.train(alpha=0.1)
    t1 = time.time()
    save_dir = os.path.join('saved_models', hp)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    with open(os.path.join(save_dir, f'jointGES-group_size-{group_size}.pkl'), 'wb') as handle:
        pickle.dump([t1-t0, A], handle)
