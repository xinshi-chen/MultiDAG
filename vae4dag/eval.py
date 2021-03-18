import torch
import math
import numpy as np
from vae4dag.common.consts import DEVICE
import os
import pickle as pkl
from vae4dag.dag_utils import is_dag, project_to_dag


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
    def eval(encoder, decoder, database, phase, k, verbose=False):

        with torch.no_grad():
            X, nll = database.static_data[phase]
            X_in, true_nll_in = X[:, :k, :], nll[:, :k]
            X_eval, true_nll_eval = X[:, k:, :], nll[:, k:]

            W = encoder(X_in.to(DEVICE))

            W = Eval.project_W(W, DEVICE, verbose)

            nll_in = torch.sum(decoder.NLL(W, X_in.to(DEVICE)), dim=-1)
            nll_eval = torch.sum(decoder.NLL(W, X_eval.to(DEVICE)), dim=-1)  # [m, n-k]

        true_nll_in, true_nll_eval = true_nll_in.mean().item(), true_nll_eval.mean().item()
        nll_in, nll_eval = nll_in.mean().item(), nll_eval.mean().item()

        if verbose:
            print('*** On observed samples ***')
            print('NLL true: %.3f, estimated: %.3f' % (true_nll_in.mean(), nll_in.mean()))
            print('*** On test samples ***')
            print('NLL true: %.3f, estimated: %.3f' % (true_nll_eval.mean(), nll_eval.mean()))

        return true_nll_in, nll_in, true_nll_eval, nll_eval

    @staticmethod
    def project_W(W, device, verbose):

        for i in range(W.shape[0]):
            if not is_dag(W[i].cpu().numpy()):
                if verbose:
                    print('%d-th W is not DAG' % i)
                W_i, _ = project_to_dag(W[i].cpu().numpy(), max_iter=50)
                W[i] = torch.tensor(W_i).to(device)
        return W
