import torch
import torch.nn as nn
import numpy as np
from vae4dag.dag_utils import sampler, project_to_dag
from vae4dag.common.utils import MLP, weights_init
import os
import pickle as pkl
from tqdm import tqdm


class Dataset(object):
    """
    synthetic dataset
    """
    def __init__(self, d, W_sparsity, W_threshold, f_hidden_dims, f_act, g_hidden_dims=None, g_act=None, verbose=True,
                 num_dags={}):
        """
        :param d: dimension of random variable
        :param W_sparsity, W_threshold: hyperparameters for generating W
        :param num_dags: number of DAGs for training (not observed)
        :param num_samples: number of samples observed from each training DAG

        X_j = f_j(Pa(X_j)) + Z * g_j(Pa(X_j))
        """

        self.d = d
        self.W_sparsity = W_sparsity
        self.W_threshold = W_threshold
        self.num_dags = num_dags
        # self.num_samples = num_samples
        self.hp = 'd-%d-ts-%.2f-sp-%.2f' % (self.d, self.W_threshold, self.W_sparsity)

        # ---------------------
        #  Load Meta Distribution
        # ---------------------

        self.data_dir = '../data'
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
        self.data_pkl = self.data_dir + '/' + self.hp + '-meta.pkl'

        if os.path.isfile(self.data_pkl):
            with open(self.data_pkl, 'rb') as f:
                self.W_mean, self.W_sd = pkl.load(f)
        else:
            # Meta Distribution of W
            self.W_mean, self.W_sd = random_W_V(d)
            with open(self.data_pkl, 'wb') as f:
                pkl.dump([self.W_mean, self.W_sd], f)

        if verbose:
            print('*** Mean of Meta Distribution ***')
            print(self.W_mean)
            print('*** SD of Meta Distribution ***')
            print(self.W_sd)

        # ---------------------
        # generate DAGs for train, validation, test
        # ---------------------

        self.static_dag = self.get_static_dags()

        # ---------------------
        #  Load Likelihood function
        # ---------------------

        self.f_hp = "-".join([f_hidden_dims, f_act])
        self.f = self.load_likelihood_function(f_hidden_dims, f_act, name='mean')

        if g_hidden_dims is not None:
            self.g_hp = "-".join([g_hidden_dims, g_act])
            self.g = self.load_likelihood_function(g_hidden_dims, g_act, name='sd')
        else:
            self.g_hp = "0"
            self.g = None

        # ---------------------
        #  Generate Observations From DAGs for Train, Validation, Test
        # ---------------------

        self.static_data = self.get_static_data()

    def load_likelihood_function(self, hidden_dims, act, name):
        dim_list = tuple(map(int, hidden_dims.split("-")))
        assert dim_list[-1] == 1

        f = []
        for i in range(self.d):
            f.append(MLP(input_dim=self.d, hidden_dims=hidden_dims, nonlinearity=act, act_last=None))
        f = nn.ModuleList(f)
        arch_hp = "-".join([name, hidden_dims, act])
        filename = self.data_dir + '/' + self.hp + '-' + arch_hp + '.dump'

        if os.path.isfile(filename):
            f.load_state_dict(torch.load(filename))
        else:
            weights_init(f)
            torch.save(f.state_dict(), filename)
        return f

    def gen_dags(self, m):
        """
        :param m: number of DAGs
        :return: DAGs represented by matrix W
        """
        W = np.random.normal(size=(m, self.d, self.d)).astype(np.float32)
        W = W * self.W_sd
        W = W + self.W_mean

        # project to DAGs sequentially
        progress_bar = tqdm(range(m))
        for i in progress_bar:
            num_skipped = 0
            while True:
                w_dag, _ = project_to_dag(W[i], sparsity=self.W_sparsity, w_threshold=self.W_threshold, max_iter=30,
                                          h_tol=1e-3, rho_max=1e+16)
                if w_dag is None:
                    # resample W
                    W[i] = np.random.normal(size=(self.d, self.d)).astype(np.float32)
                    W[i] = W[i] * self.W_sd
                    W[i] = W[i] + self.W_mean
                    num_skipped += 1
                    progress_bar.set_description('skipped: %d' % num_skipped)
                else:
                    W[i] = w_dag
                    break
        return W

    def get_static_dags(self):
        static_dag = dict()
        for phase in self.num_dags:
            data_pkl = self.data_dir + '/' + self.hp + '-%s-dag-%d.pkl' % (phase, self.num_dags[phase])
            if os.path.isfile(data_pkl):
                with open(data_pkl, 'rb') as f:
                    static_dag[phase] = pkl.load(f)
            else:
                static_dag[phase] = self.gen_dags(self.num_dags[phase])
                with open(data_pkl, 'wb') as f:
                    pkl.dump(static_dag[phase], f)
        return static_dag

    def get_static_data(self):
        pass

    def gen_batch_sample(self, W, n):

        assert len(W.shape) == 3

        if isinstance(W, np.ndarray):
            W = torch.tensor(W)

        num_dags = W.shape[0]
        X = torch.zeros(size=[num_dags, n, self.d])
        nll = torch.zeros(size=[num_dags, n])
        for i in range(num_dags):
            X[i, :, :], nll[i] = sampler(W[i], n, self.f, self.g)
        return X.detach(), nll.detach()

    def load_data(self, batch_size, auto_reset=False, shuffle=True, device=None, phase='train'):

        if phase != 'train':
            assert not shuffle
            assert not auto_reset
        X, nll = self.static_data[phase]
        num_dags = X.shape[0]
        idx = torch.arange(0, num_dags)

        while True:
            if shuffle:
                perms = torch.randperm(num_dags)
                X = X[perms, :, :]
                idx = idx[perms]
                nll = nll[perms]

            for pos in range(0, num_dags, batch_size):
                if pos + batch_size > num_dags:  # the last mini-batch has fewer samples
                    if auto_reset:  # no need to use this last mini-batch
                        break
                    else:
                        num_samples = num_dags - pos
                else:
                    num_samples = batch_size
                if device is None:
                    yield X[pos : pos + num_samples, :, :].detach(), idx[pos : pos + num_samples].detach(),\
                          nll[pos : pos + num_samples].detach()
                else:
                    yield X[pos : pos + num_samples, :, :].detach().to(device), idx[pos : pos + num_samples].\
                        detach().to(device), nll[pos : pos + num_samples].detach().to(device)
            if not auto_reset:
                break

    @staticmethod
    def shuffle_order_of_sample(X, nll, device=None):
        assert len(X.shape) == 3
        n = X.shape[1]
        assert n == nll.shape[1]
        perms = torch.randperm(n)
        if device is not None:
            perms = perms.to(device)
        return X[:, perms, :].detach(), nll[:, perms].detach()


class GenDataset(Dataset):
    def __init__(self, d, n, W_sparsity, W_threshold, f_hidden_dims, f_act, g_hidden_dims=None, g_act=None, verbose=True,
                 num_dags={}, num_test=None):

        self.n = n
        if num_test is None:
            self.num_test = n
        else:
            self.num_test = num_test

        super(GenDataset, self).__init__(d, W_sparsity, W_threshold, f_hidden_dims, f_act, g_hidden_dims, g_act, verbose,
                                         num_dags)

    def get_static_data(self):

        dataset_hp = [self.hp, self.f_hp, self.g_hp]
        dataset_hp += list(self.num_dags.values())
        dataset_hp += [self.n, self.num_test]
        self.dataset_hp = "-".join(map(str, dataset_hp))

        static_data = dict()

        for phase in self.num_dags:

            data_pkl = self.data_dir + '/' + self.dataset_hp + '-%s.pkl' % phase

            if os.path.isfile(data_pkl):
                with open(data_pkl, 'rb') as f:
                    static_data[phase] = pkl.load(f)
            else:
                if phase != 'test':
                    num_samples = self.n
                else:
                    num_samples = self.num_test
                static_data[phase] = self.gen_batch_sample(W=self.static_dag[phase],
                                                           n=num_samples)
                with open(data_pkl, 'wb') as f:
                    pkl.dump(static_data[phase], f)

        return static_data


def random_W_V(d, p=0.5):
    # mean
    A_int = (np.random.rand(d, d) < p)
    A = A_int.astype(np.float32)
    U = np.random.uniform(low=5.0, high=20.0, size=[d, d])
    U[np.random.rand(d, d) < 0.5] *= -1
    W = A * U.astype(np.float32)

    # sd
    SD = np.random.rand(d, d).astype(np.float32) * 5
    # if mean is zero, then make SD small (unlikely to have an edge)
    SD = SD * 0.05 + (SD * 0.95) * A
    return W, SD


if __name__ == '__main__':

    d = 5
    num_dags = 5
    num_sample = 5
    threshold = 0.1
    sparsity = 1.0

    dataset = GenDataset(d, num_sample, sparsity, threshold, f_hidden_dims='6-1', f_act='relu',
                         num_dags={'train': num_dags, 'vali': num_dags, 'test': num_dags}, num_test=num_sample)
    batch_size = 10
    from vae4dag.common.consts import DEVICE
    data_loader = dataset.load_data(batch_size=10, device=DEVICE)

    for dag in dataset.static_dag['train']:
        print(dag)

    iterations = len(range(0, num_dags, batch_size))
    for i in range(iterations):
        data = next(data_loader)
        x, idx, nll = data
        print(i)
        print(x)
