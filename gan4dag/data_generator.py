import torch
import numpy as np
from gan4dag.dag_utils import sampler, is_dag, project_to_dag


class LsemDataset(object):
    """
    synthetic dataset
    Linear SEM
    """
    def __init__(self, W_mean, W_sd, W_sparsity, W_threshold, noise_mean, noise_sd, num_dags, num_sample):
        """
        :param W_mean: mean of W_ij
        :param W_sd: sd of W_ij
        :param noise_mean: Z ~ N(noise_mean, noise_sd)
        :param noise_sd: Z ~ N(noise_mean, noise_sd)
        :param num_dags: number of DAGs for training (not observed)
        :param num_sample: number of samples observed from each training DAG
        """

        self.d = W_mean.shape[0]

        # Distribution Of Noise
        self.noise_mean = noise_mean
        self.noise_sd = noise_sd

        # Meta Distribution of W
        self.W_mean = W_mean
        self.W_sd = W_sd
        self.W_sparsity = W_sparsity
        self.W_threshold = W_threshold

        # Generate static data for training
        self.num_sample = num_sample
        self.num_dags = num_dags

        self.train_data = dict()
        self.train_data['dag'] = self.gen_dags(num_dags)
        self.train_data['data'] = self.gen_batch_sample(W=self.train_data['dag'],
                                                        n=self.num_sample)

    def gen_dags(self, m):
        """
        :param m: number of DAGs
        :return: DAGs represented by matrix W
        """
        W = np.random.normal(size=(m, self.d, self.d))
        W = W * self.W_sd
        W = W + self.W_mean

        # project to DAGs sequentially
        for i in range(m):
            while True:
                w_dag, _ = project_to_dag(W[i], sparsity=self.W_sparsity, w_threshold=self.W_threshold,
                                          max_iter=10, h_tol=1e-3, rho_max=1e+16)
                if w_dag is None:
                    # resample W
                    W[i] = np.random.normal(size=(self.d, self.d))
                    W[i] = W[i] * self.W_sd
                    W[i] = W[i] + self.W_mean
                else:
                    W[i] = w_dag
                    break
        return W

    def gen_batch_sample(self, W, n):
        assert len(W.shape) == 3
        num_dags = W.shape[0]
        X = np.zeros([num_dags, n, self.d])
        for i in range(num_dags):
            X[i, :, :] = sampler(W[i], n, self.noise_mean, self.noise_sd, noise_type='gauss')
        return X

    def load_data(self, batch_size, auto_reset=False, shuffle=True, device=None):

        X = torch.tensor(self.train_data['data'])
        while True:
            if shuffle:
                perms = torch.randperm(self.num_dags)
                X = X[perms, :]
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
    from gan4dag.common.consts import DEVICE
    data_loader = dataset.load_data(batch_size=10, device=DEVICE)

    iterations = len(range(0, num_dags, batch_size))
    for i in range(iterations):
        x = next(data_loader)
        print(i)
        print(x)
