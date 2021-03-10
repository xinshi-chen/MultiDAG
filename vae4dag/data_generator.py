import torch
import numpy as np
from vae4dag.dag_utils import sampler, is_dag, project_to_dag, run_notears_linear
import os
import pickle as pkl
from tqdm import tqdm



class LsemDataset(object):
    """
    synthetic dataset
    Linear SEM
    """
    def __init__(self, d, W_sparsity, W_threshold, num_dags, num_sample, verbose=True):
        """
        :param d: dimension of random variable
        :param W_sparsity, W_threshold: hyperparameters for generating W
        :param num_dags: number of DAGs for training (not observed)
        :param num_sample: number of samples observed from each training DAG
        """

        self.d = d
        self.W_sparsity = W_sparsity
        self.W_threshold = W_threshold

        self.hp = 'LSEM-d-%d-ts-%.2f-sp-%.2f' % (self.d, self.W_threshold, self.W_sparsity)

        # ---------------------
        #  Load Meta Distribution
        # ---------------------

        self.data_dir = '../data'
        if not os.path.isdir(self.data_dir):
            os.makedirs(self.data_dir)
        self.data_pkl = self.data_dir + '/' + self.hp + '-meta.pkl'

        if os.path.isfile(self.data_pkl):
            with open(self.data_pkl, 'rb') as f:
                self.W_mean, self.W_sd, self.noise_mean, self.noise_sd = pkl.load(f)
        else:
            # Meta Distribution of W
            self.W_mean = np.random.normal(size=[d, d]).astype(np.float32) * 3
            self.W_sd = np.random.rand(d, d).astype(np.float32)
            # Distribution Of Noise
            self.noise_mean = np.zeros(d, dtype=np.float32)
            self.noise_sd = np.ones(d, dtype=np.float32)
            with open(self.data_pkl, 'wb') as f:
                pkl.dump([self.W_mean, self.W_sd, self.noise_mean, self.noise_sd], f)

        if verbose:
            print('*** Mean of Meta Distribution ***')
            print(self.W_mean)
            print('*** SD of Meta Distribution ***')
            print(self.W_sd)

        # ---------------------
        #  Static Data For Training
        # ---------------------
        self.num_sample = num_sample
        self.num_dags = num_dags
        self.train_data = dict()

        # generate DAGs
        data_pkl = self.data_dir + '/' + self.hp + '-train-dag-%d.pkl' % num_dags
        if os.path.isfile(data_pkl):
            with open(data_pkl, 'rb') as f:
                self.train_data['dag'] = pkl.load(f)
        else:
            self.train_data['dag'] = self.gen_dags(num_dags)
            with open(data_pkl, 'wb') as f:
                pkl.dump(self.train_data['dag'], f)

        # generate observed data
        data_pkl = self.data_dir + '/' + self.hp + '-train-data-%d-%d.pkl' % (num_dags, num_sample)
        if os.path.isfile(data_pkl):
            with open(data_pkl, 'rb') as f:
                self.train_data['data'] = pkl.load(f)
        else:
            self.train_data['data'] = self.gen_batch_sample(W=self.train_data['dag'],
                                                            n=self.num_sample)
            with open(data_pkl, 'wb') as f:
                pkl.dump(self.train_data['data'], f)

        self.static = dict()

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
            while True:
                w_dag, _ = project_to_dag(W[i], sparsity=self.W_sparsity, w_threshold=self.W_threshold, max_iter=10,
                                          h_tol=1e-3, rho_max=1e+16)
                if w_dag is None:
                    # resample W
                    W[i] = np.random.normal(size=(self.d, self.d)).astype(np.float32)
                    W[i] = W[i] * self.W_sd
                    W[i] = W[i] + self.W_mean
                else:
                    W[i] = w_dag
                    break
        return W

    def gen_batch_sample(self, W, n):
        assert len(W.shape) == 3
        num_dags = W.shape[0]
        X = np.zeros([num_dags, n, self.d], dtype=np.float32)
        for i in range(num_dags):
            X[i, :, :] = sampler(W[i], n, self.noise_mean, self.noise_sd, noise_type='gauss')
        return X

    def load_data(self, batch_size, auto_reset=False, shuffle=True, device=None, baseline=False):

        if baseline:
            X = torch.tensor(self.static['to-dag'])
        else:
            X = torch.tensor(self.train_data['data'])

        while True:
            if shuffle:
                perms = torch.randperm(self.num_dags)
                X = X[perms, :, :]
            for pos in range(0, self.num_dags, batch_size):
                if pos + batch_size > self.num_dags:  # the last mini-batch has fewer samples
                    if auto_reset:  # no need to use this last mini-batch
                        break
                    else:
                        num_samples = self.num_dags - pos
                else:
                    num_samples = batch_size
                if device is None:
                    yield X[pos : pos + num_samples, :, :].detach()
                else:
                    yield X[pos : pos + num_samples, :, :].detach().to(device)
            if not auto_reset:
                break

    def train_to_dag(self):

        filename = self.data_dir + '/' + self.hp + '-train-data-%d-%d-to-DAG.pkl' % (self.num_dags, self.num_sample)
        # load if exists
        if os.path.isfile(filename):
            with open(filename, 'rb') as f:
                W_est = pkl.load(f)
        else:
            # Run NOTEARS
            W_est = run_notears_linear(self.train_data['data'])
            with open(filename, 'wb') as f:
                pkl.dump(W_est, f)

        self.static['to-dag'] = W_est


if __name__ == '__main__':
    d = 5
    num_dags = 30
    num_sample = 5
    threshold = 0.1
    sparsity = 1.0

    W_mean = np.random.normal(size=[d, d]) * 3
    W_sd = np.random.rand(d, d)

    noise_mean = np.zeros(d)
    noise_sd = np.ones(d)

    dataset = LsemDataset(W_mean, W_sd, sparsity, threshold, noise_mean, noise_sd, num_dags, num_sample)
    batch_size = 10
    from vae4dag.common.consts import DEVICE
    data_loader = dataset.load_data(batch_size=10, device=DEVICE)

    iterations = len(range(0, num_dags, batch_size))
    for i in range(iterations):
        x = next(data_loader)
        print(i)
        print(x)
