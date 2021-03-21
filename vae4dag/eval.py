import torch
import math
import numpy as np
from vae4dag.common.consts import DEVICE
import os
import pickle as pkl
from vae4dag.dag_utils import is_dag, project_to_dag
from notears.utils import count_accuracy


class Eval:
    def __init__(self, encoder, decoder, database, save_dir, model_dump):
        self.db = database
        self.save_dir = save_dir
        self.model_dump = model_dump
        self.encoder = encoder
        self.decoder = decoder
        self.load()
        self.encoder.eval()
        self.decoder.eval()

    def load(self):
        dump = self.save_dir + '/best-' + self.model_dump
        self.encoder.load_state_dict(torch.load(dump))

        dump = dump[:-5] + '_decoder.dump'
        self.decoder.load_state_dict(torch.load(dump))

    @staticmethod
    def eval(encoder, decoder, database, phase, k, verbose=False, return_W=False):

        with torch.no_grad():
            X, nll = database.static_data[phase]
            X_in, true_nll_in = X[:, :k, :], nll[:, :k]
            X_eval, true_nll_eval = X[:, k:, :], nll[:, k:]

            W = encoder(X_in.to(DEVICE))
            if verbose:
                print(W)
            W = Eval.project_W(W, DEVICE, verbose)
            if verbose:
                print(W)
            nll_in = torch.sum(decoder.NLL(W, X_in.to(DEVICE)), dim=-1)
            nll_eval = torch.sum(decoder.NLL(W, X_eval.to(DEVICE)), dim=-1)  # [m, n-k]

        true_nll_in, true_nll_eval = true_nll_in.mean().item(), true_nll_eval.mean().item()
        nll_in, nll_eval = nll_in.mean().item(), nll_eval.mean().item()

        if verbose:
            print('*** On observed samples ***')
            print('NLL true: %.3f, estimated: %.3f' % (true_nll_in, nll_in))
            print('*** On test samples ***')
            print('NLL true: %.3f, estimated: %.3f' % (true_nll_eval, nll_eval))
        if return_W:
            return true_nll_in, nll_in, true_nll_eval, nll_eval, W
        else:
            return true_nll_in, nll_in, true_nll_eval, nll_eval

    @staticmethod
    def project_W(W, device, verbose, w_threshold=0.01, sparsity=0.1):
        if not isinstance(W, np.ndarray):
            W = W.detach().cpu().numpy()

        for i in range(W.shape[0]):
            if not is_dag(W[i]):
                if verbose:
                    print('%d-th W is not DAG' % i)
                W_i, _ = project_to_dag(W[i], max_iter=50, w_threshold=w_threshold, sparsity=sparsity)
                W[i] = W_i
        if device is None:
            return W
        else:
            return torch.tensor(W).to(device)


def eval_structure_1pair(W: np.ndarray, W_true: np.ndarray):
    results = count_accuracy(np.abs(W_true)>1e-15, (np.abs(W)>1e-15))
    results['mse'] = np.sqrt(((W_true - W) ** 2).sum())
    return results


def eval_structure(W, W_true):

    if torch.is_tensor(W):
        W = W.detach().cpu().numpy()
    if torch.is_tensor(W_true):
        W_true = W_true.cpu().numpy()

    result = dict()
    for i in range(W_true.shape[0]):
        result_i = eval_structure_1pair(W[i], W_true[i])
        for key in result_i:
            if key not in result:
                result[key] = []
            result[key].append(result_i[key])
    return result
