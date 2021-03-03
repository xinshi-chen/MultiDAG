
from notears.linear import notears_linear
from gan4dag.common.consts import DEVICE, OPTIMIZER
from gan4dag.trainer import LsemTrainer
from gan4dag.eval import Eval
from gan4dag.common.cmd_args import cmd_args
from gan4dag.data_generator import LsemDataset
from gan4dag.dag_utils import is_dag
import random
import numpy as np
from gan4dag.gan_model import GenNet, DiscGIN
import torch
from tqdm import tqdm
import os
import pickle as pkl


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

    # ---------------------
    #  Synthetic Dataset
    # ---------------------

    print('*** Loading data ***')
    db = LsemDataset(d, sparsity, threshold, num_dag, num_sample)
    print('*** Data loaded ***')

    # ---------------------
    #  Observations -> DAGs (Run NOTEARS)
    # ---------------------

    filename = db.data_dir + '/' + db.hp + '-train-data-%d-%d-to-DAG.pkl' % (num_dag, num_sample)
    if os.path.isfile(filename):
        with open(filename, 'rb') as f:
            W_est = pkl.load(f)
    else:
        # Run NOTEARS
        X = db.train_data['data']
        W_est = np.zeros([num_dag, d, d])
        progress_bar = tqdm(range(num_dag))
        for i in progress_bar:
            W_est[i] = notears_linear(X[i], lambda1=0.1, loss_type='l2')
            assert is_dag(W_est[i])

        with open(filename, 'wb') as f:
            pkl.dump(W_est, f)

    db.static['to-dag'] = W_est.astype(np.float32)

    # ---------------------
    #  DAGs -> generative model
    # ---------------------
    #  (1) Initialize Networks
    # ---------------------

    print('*** Initializing networks ***')

    hidden_dim = '32-32'
    output_hidden_dim = '64-1'
    act = output_act = 'relu'

    hp_arch = 'f-%s-%s-out-%s-%s' % (hidden_dim, act, output_hidden_dim, output_act)
    hp_train = 'm-%d-n-%d-gen-%d-ep-%d-bs-%d-glr-%.5f-dlr-%.5f' % (cmd_args.num_dag, cmd_args.num_sample,
                                                                   num_sample_gen, cmd_args.num_epochs,
                                                                   cmd_args.batch_size, cmd_args.g_lr, cmd_args.d_lr)
    model_dump = 'baseline' + hp_arch + '-' + hp_train + '.dump'

    if cmd_args.learn_noise:
        noise_mean, noise_sd = None, None
    else:
        noise_mean, noise_sd = db.noise_mean, db.noise_sd

    gen_net = GenNet(d=d,
                     noise_mean=noise_mean,
                     noise_sd=noise_sd).to(DEVICE)
    disc_net = DiscGIN(d=d,
                       hidden_dims=hidden_dim,
                       nonlinearity=act,
                       output_hidden_dims=output_hidden_dim,
                       output_nonlinearity=output_act).to(DEVICE)

    if cmd_args.phase == 'train':
        # ---------------------
        #  (2) Optimizer
        # ---------------------

        g_opt = OPTIMIZER[cmd_args.g_optimizer](gen_net.parameters(),
                                                lr=cmd_args.g_lr,
                                                weight_decay=cmd_args.weight_decay)
        d_opt = OPTIMIZER[cmd_args.d_optimizer](disc_net.parameters(),
                                                lr=cmd_args.d_lr,
                                                weight_decay=cmd_args.weight_decay)

        # ---------------------
        #  (3) Trainer
        # ---------------------
        trainer = LsemTrainer(gen_net, disc_net, g_opt, d_opt, db, num_sample_gen=num_sample_gen, save_dir=cmd_args.save_dir,
                              model_dump=model_dump, save_itr=cmd_args.save_itr)
        trainer.train(epochs=cmd_args.num_epochs, batch_size=cmd_args.batch_size, baseline=True)

    if cmd_args.phase == 'test':
        # ---------------------
        #  Eval
        # ---------------------
        evaluator = Eval(database=db, save_dir=cmd_args.save_dir, model_dump=model_dump, save_itr=cmd_args.save_itr)

        result = evaluator.eval(gen_net, m_small=128, m_large=2048,  verbose=True, bw=1.0)
        # print('mmd: ')
        # print(result['mmd'][1])
        print('ce: ')
        print(result['ce'][1])
        print('parameter: ')
        print(result['parameter'][1])
