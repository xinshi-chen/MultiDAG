
from notears.linear import notears_linear
from gan4dag.common.consts import DEVICE, OPTIMIZER
from gan4dag.trainer import LsemTrainer
from gan4dag.eval import Eval
from gan4dag.common.cmd_args import cmd_args
from gan4dag.data_generator import LsemDataset
from gan4dag.dag_utils import is_dag
import random
import numpy as np
from gan4dag.gan_model import GenNet, DiscNet
import torch
from tqdm import tqdm


if __name__ == '__main__':
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    # ---------------------
    #  Hyperparameters
    # ---------------------

    num_dag = cmd_args.num_dag
    num_sample = cmd_args.num_sample
    num_sample_gen = cmd_args.num_sample_gen
    threshold = cmd_args.threshold
    sparsity = cmd_args.sparsity
    d = cmd_args.d

    hp_arch = 'f-%s-%s-out-%s-%s' % (cmd_args.f_hidden_dim, cmd_args.f_act, cmd_args.output_hidden_dim,
                                     cmd_args.output_act)
    hp_train = 'm-%d-n-%d-gen-%d-ep-%d-bs-%d-glr-%.5f-dlr-%.5f' % (cmd_args.num_dag, cmd_args.num_sample,
                                                                   num_sample_gen, cmd_args.num_epochs,
                                                                   cmd_args.batch_size, cmd_args.g_lr, cmd_args.d_lr)
    model_dump = hp_arch + '-' + hp_train + '.dump'

    # ---------------------
    #  Synthetic Dataset
    # ---------------------

    print('*** Loading data ***')
    db = LsemDataset(d, sparsity, threshold, num_dag, num_sample)

    # ---------------------
    #  Observations -> DAGs (notears)
    # ---------------------

    X = db.train_data['data']
    W_est = np.zeros([num_dag, d, d])
    progress_bar = tqdm(range(num_dag))
    for i in progress_bar:
        W_est[i] = notears_linear(X[i], lambda1=0.1, loss_type='l2')
        assert is_dag(W_est[i])

    # ---------------------
    #  DAGs -> generative model
    # ---------------------
