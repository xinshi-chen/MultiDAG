import torch
import numpy as np
import os
import pickle as pkl

class SergioDataset(object):
    """
    SERGIO Dataset
    """
    def __init__(self, dir):
        """
        :param dir: .npz data dir
        """
        data = np.load(dir)
        self.K = data['task_labels'].max() + 1
        self.X = data['expression']
        self.X = torch.Tensor(self.X.reshape(self.K, -1, self.X.shape[-1]))
        self.G = data['task_adjacencies']
        self.T = data['true_adjacency']
        self.n = self.X.shape[1]
        self.p = self.X.shape[2]
        self.hp = os.path.basename(dir)

    def load_data(self, batch_size, device=None):
        idx = np.random.permutation(self.X.shape[1])[:batch_size]
        if device is None:
            return self.X[:, idx]
        else:
            return self.X[:, idx].to(device)
