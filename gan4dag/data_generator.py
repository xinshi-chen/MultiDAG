import torch
import numpy as np
import networkx as nx
import random
from gan4dag.common.consts import DEVICE


def random_dag(d: int,
               degree: float,
               graph_type: str = 'ER',
               w_range: tuple = (0.5, 2.0)):
    """Simulate random DAG with some expected degree.

    Reference: modified from https://github.com/fishmoon1234/DAG-GNN/blob/master/src/utils.py

    Args:
        d: number of nodes
        degree: expected node degree, in + out
        graph_type: {erdos-renyi, barabasi-albert, full}
        w_range: weight range +/- (low, high)
    Returns:
        G: weighted DAG
    """

    if graph_type == 'ER':  # erdos-renyi
        prob = float(degree) / (d - 1)
        B = np.tril((np.random.rand(d, d) < prob).astype(float), k=-1)
    elif graph_type == 'BA':  # barabasi-albert
        m = int(round(degree / 2))
        B = np.zeros([d, d])
        bag = [0]
        for ii in range(1, d):
            dest = np.random.choice(bag, size=m)
            for jj in dest:
                B[ii, jj] = 1
            bag.append(ii)
            bag.extend(dest)
    elif graph_type == 'full':  # ignore degree, only for experimental use
        B = np.tril(np.ones([d, d]), k=-1)
    else:
        raise ValueError('unknown graph type')

    # random permutation
    # TODO: biased permutation
    P = np.random.permutation(np.eye(d, d))  # permutes first axis only
    B_perm = P.T.dot(B).dot(P)
    U = np.random.uniform(low=w_range[0], high=w_range[1], size=[d, d])
    U[np.random.rand(d, d) < 0.5] *= -1
    W = (B_perm != 0).astype(float) * U
    G = nx.DiGraph(W)
    return G


def _sample_noise(mu, sigma, size, noise_type='gauss'):
    if noise_type == 'gauss':
        z = np.random.normal(size=size)
        z *= sigma
        z += mu
    else:
        raise ValueError('unknown noise type')
    return z


def sample_lsem(G: nx.DiGraph,
                n: int,
                noise_mu: np.ndarray,
                noise_sigma: np.ndarray) -> np.ndarray:
    """Simulate samples from SEM with specified type of noise.

    Reference: modified from https://github.com/fishmoon1234/DAG-GNN/blob/master/src/utils.py

    Args:
        G: weigthed DAG
        n: number of samples
        noise_mu: mean of noise distribution
        noise_sigma: sd of noise distribution
    Returns:
        X: [n,d] sample matrix
    """

    W = nx.to_numpy_array(G)
    d = W.shape[0]
    assert len(noise_mu) == d
    assert len(noise_sigma) == d
    X = np.zeros([n, d])
    ordered_vertices = list(nx.topological_sort(G))
    assert len(ordered_vertices) == d
    for j in ordered_vertices:
        parents = list(G.predecessors(j))
        eta = X[:, parents].dot(W[parents, j])
        noise = _sample_noise(noise_mu[j], noise_sigma[j], n)
        X[:, j] = eta + noise

    return X


class Dataset(object):
    """
    synthetic dataset
    Linear SEM
    """
    def __init__(self, d, W_bias, noise_mu=None, noise_sigma=None):
        """
        :param d: dimension of random variable
        :param W_bias:
        """

        self.d = d

        # Distribution Of Noise
        if noise_mu is None:
            noise_mu = np.random.rand(d)  # in [0, 1)
        else:
            assert len(noise_mu) == d

        if noise_sigma is None:
            noise_sigma = np.random.rand(d) + 1  # in [1, 2)
        else:
            assert len(noise_sigma) == d

        self.dim_in = args.dim_in
        self.u_inf = args.u_inf
        self.u_sup = args.u_sup
        self.b_inf = args.b_inf
        self.b_sup = args.b_sup
        self.static_data = dict()
        self.energy = true_energy
        self.energy.eval()
        self.num_test = population
        self.num_train = train
        u, q, b, x = self.get_samples(population)
        # find label
        self.static_data['test'] = (u, q, b, x)
        self.static_data['train'] = (u[:train], q[:train], b[:train], x[:train])

    def resample_train(self, seed):
        g = torch.Generator()
        g.manual_seed(seed)
        perms = torch.randperm(self.num_test, generator=g)[:self.num_train].to(DEVICE)
        u, q, b, x = self.static_data['test']
        self.static_data['train'] = u[perms, :], q[perms, :], b[perms, :], x[perms, :]

    def get_samples(self, size):

        u, q, b = self.get_u_q_b(size)
        # label
        with torch.no_grad():
            e = self.energy.forward(u)
            w_inv = qeq_inv(q, e)
            x = torch.einsum('bij,bj->bi', w_inv, b)

        return u, q, b, x.detach()

    def get_u_q_b(self, size):
        # u
        u = torch.rand([size, self.dim_in]).to(DEVICE) * (self.u_sup - self.u_inf) + self.u_inf
        b = torch.rand([size, self.d]).to(DEVICE) * (self.b_sup - self.b_inf) + self.b_inf

        # q -> random orthogonal matrix
        orth_q = torch.tensor(batch_orthogonal(size, self.d)).float().to(DEVICE)

        return u, orth_q, b

    def load_data(self, batch_size, phase, auto_reset=True, shuffle=True):

        assert phase in self.static_data

        u, q, b, x = self.static_data[phase]
        data_size = u.shape[0]
        while True:
            if shuffle:
                perms = torch.randperm(data_size)
                u = u[perms, :]
                q = q[perms, :]
                b = b[perms, :]
                x = x[perms, :]
            for pos in range(0, data_size, batch_size):
                if pos + batch_size > data_size:  # the last mini-batch has fewer samples
                    if auto_reset:  # no need to use this last mini-batch
                        break
                    else:
                        num_samples = data_size - pos
                else:
                    num_samples = batch_size

                yield u[pos : pos + num_samples, :], q[pos : pos + num_samples, :], b[pos : pos + num_samples, :], x[pos : pos + num_samples, :]
            if not auto_reset:
                break

if __name__ == '__main__':
    random.seed(101)
    np.random.seed(101)
    torch.manual_seed(101)

    d = 5
    degree = 2
    g = random_dag(d, degree)
    mu = np.array([0, 1.0, 2.0, 3.0, 4.0])
    sigma = np.ones(5)
    x = sample_lsem(g, 3, mu, sigma)
    print(x)




