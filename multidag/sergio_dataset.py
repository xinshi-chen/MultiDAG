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
        self.labels = data['task_labels']
        self.K = data['task_labels'].max() + 1
        self.X_raw = data['expression']
        self.X = torch.Tensor(self.X_raw.reshape(self.K, -1, self.X_raw.shape[-1]))
        self.G = data['task_adjacencies']
        self.T = np.expand_dims(data['true_adjacency'], 0)
        self.n = self.X.shape[1]
        self.p = self.X.shape[2]
        self.hp = os.path.basename(dir)

    def load_data(self, batch_size, device=None):
        idx = np.random.permutation(self.X.shape[1])[:batch_size]
        if device is None:
            return self.X[:, idx]
        else:
            return self.X[:, idx].to(device)

    def load_by_task(self):
        data_by_task = []
        for task in range(self.K):
            idx = self.labels == task
            data_by_task.append(self.X_raw[idx])
        return data_by_task

