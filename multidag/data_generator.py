import torch
import numpy as np
import os
import pickle as pkl
import networkx as nx


class Dataset(object):
    """
    synthetic dataset
    """
    def __init__(self, p, n, K, s0, s, d, w_range: tuple = (0.5, 2.0), verbose=True):
        """
        :param p: dimension of random variable
        :param n: number of samples per DAG
        :param K: number of DAGs
        :param s: sparsity of union support
        :param s0: sparsity of each DAG
        :param d: max number of parents
        :param w_range: range of edge weights
        """

        # hyper-parameters
        self.p = p
        self.n = n
        self.K = K
        self.s0 = s0
        self.s = s
        self.d = d
        self.w_range = w_range

        hp_dict = {'p': p,
                   'n': n,
                   'K': K,
                   's': s,
                   's0': s0,
                   'd': d,
                   'w_range_l': w_range[0],
                   'w_range_u': w_range[1]}

        self.hp = ''
        for key in hp_dict:
            self.hp += key + '-' + str(hp_dict[key]) + '_'
        self.hp = self.hp[:-1]

        # ---------------------
        #  Load DAGs
        # ---------------------

        if verbose:
            print('*** Loading DAGs ***')

        self.data_dir = os.path.join('../data', self.hp)
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
        data_pkl = self.data_dir + f'/DAGs.pkl'

        if os.path.isfile(data_pkl):
            with open(data_pkl, 'rb') as f:
                self.G, self.Perm = pkl.load(f)
        else:
            self.G, self.Perm = random_G(self.p, self.s, self.s0, self.d, self.K, self.w_range)
            with open(data_pkl, 'wb') as f:
                pkl.dump([self.G, self.Perm], f)

        # ---------------------
        #  Load Samples From DAGs
        # ---------------------
        if verbose:
            print('*** Loading Samples ***')

        data_pkl = self.data_dir + f'/samples_gid.pkl'

        if os.path.isfile(data_pkl):
            with open(data_pkl, 'rb') as f:
                self.X = pkl.load(f)
        else:
            self.X = self.gen_samples(self.G, self.n)
            with open(data_pkl, 'wb') as f:
                pkl.dump(self.X, f)

    def gen_samples(self, G, n):

        assert len(G.shape) == 3

        if isinstance(G, np.ndarray):
            G = torch.tensor(G)

        num_dags = G.shape[0]
        X = torch.zeros(size=[num_dags, n, self.p])
        for i in range(num_dags):
            X[i, :, :] = sampler(G[i], n)
        return X.detach()

    def load_batch_data(self, batch_size, auto_reset=False, shuffle=True, device=None, phase='train'):

        if phase != 'train':
            assert not shuffle
            assert not auto_reset
        num_dags = self.X.shape[0]
        idx = torch.arange(0, num_dags)
        X = self.X

        while True:
            if shuffle:
                perms = torch.randperm(num_dags)
                X = X[perms, :, :]
                idx = idx[perms]

            for pos in range(0, num_dags, batch_size):
                if pos + batch_size > num_dags:  # the last mini-batch has fewer samples
                    if auto_reset:  # no need to use this last mini-batch
                        break
                    else:
                        num_samples = num_dags - pos
                else:
                    num_samples = batch_size
                if device is None:
                    yield X[pos: pos + num_samples, :, :].detach(), idx[pos: pos + num_samples].detach()
                else:
                    yield X[pos: pos + num_samples, :, :].detach().to(device), idx[pos: pos + num_samples].detach().to(device)
            if not auto_reset:
                break

    def load_data(self, batch_size=20, device=None):
        idx = np.random.permutation(self.X.shape[1])[:batch_size]
        if device is None:
            return self.X[:, idx]
        else:
            return self.X[:, idx].to(device)




def random_G(p, s, s0, d, K, w_range: tuple = (0.5, 2.0)):
    """
    p: dimension
    s: union sparsity
    s0: sparsity of each DAG
    d: max number of parents
    K: number of DAGs
    w_range: weight range of G_ij
    """

    B = np.tril(np.ones([p, p]), k=-1)

    # TODO:
    #  (1) Random sample submatrix S from B such that |supp(S)| = s
    #  (2) Random sample K submatrices G[k] from S such that
    #      (i) |supp(G[k])| = s_0
    #      (ii) |supp(G[k][:, j]])| <= d
    # union support
    G = np.zeros([K, p, p])
    assert K * s0 >= s

    count_g = 0
    while True:
        indices = np.where(B > 0)
        random_pos = np.random.permutation(len(indices[0]))[:s]
        S = np.zeros((p, p))
        row_indices = indices[0][random_pos]
        col_indices = indices[1][random_pos]
        S[(row_indices, col_indices)] = 1

        for k in range(K):
            count_k = 0
            while True:
                random_pos = np.random.permutation(s)[:s0]
                G[k][(row_indices[random_pos], col_indices[random_pos])] = 1
                if G[k].sum(axis=1).max() <= d:
                    break
                else:
                    G[k] = 0
                    count_k += 1
                    if count_k > 1e4:
                        raise ValueError('Please improve the sample strategy for G[k]')
        if (G.sum(axis=0) >= S).all():
            break
        else:
            count_g += 1
            if count_g > 1e4:
                raise ValueError('Please improve the sample strategy for G')

    # permutation matrix
    Perm = np.random.permutation(np.eye(p, p))

    for k in range(K):
        G_perm = Perm.T.dot(G[k]).dot(Perm)
        # weights
        U = np.random.uniform(low=w_range[0], high=w_range[1], size=[p, p])
        U[np.random.rand(p, p) < 0.5] *= -1

        G[k] = (G_perm != 0).astype(float) * U

    return G, Perm


def sampler(G, n):
    """
    sample n samples from the probablistic model defined by G

    :param G: weighted adjacency matrix. size=[p, p]
    :param n: number of samples

    :return: X: [n, p] sample matrix
    """

    if isinstance(G, np.ndarray):
        G = torch.tensor(G)
    elif not torch.is_tensor(G):
        raise NotImplementedError('Adjacency matrix should be np.ndarray or torch.tensor.')

    p = G.shape[0]

    X = torch.zeros([n, p])
    # noise
    z = torch.normal(0, 1, size=(n, p)).float()

    # get the topological order of the DAG
    # todo: check change
    # G = nx.DiGraph(G.detach().cpu().numpy())
    # ordered_vertices = list(nx.topological_sort(G))
    g = nx.DiGraph(G.detach().cpu().numpy())
    ordered_vertices = list(nx.topological_sort(g))
    assert len(ordered_vertices) == p
    for j in ordered_vertices:

        GX = G[:, j] * X  # [n, p]
        m_j = GX.sum(dim=-1)   # linear model
        X[:, j] = m_j + z[:, j]  # add noise
    return X

if __name__ == '__main__':
    from multidag.common.cmd_args import cmd_args
    db = Dataset(p=cmd_args.p,
                 n=cmd_args.n_sample,
                 K=cmd_args.K,
                 s0=cmd_args.s0,
                 s=cmd_args.s,
                 d=cmd_args.d,
                 w_range=(0.5, 2.0), verbose=True)

