import torch
from gan4dag.common.consts import DEVICE, OPTIMIZER
from gan4dag.trainer import LsemTrainer
from gan4dag.eval import Eval
from gan4dag.common.cmd_args import cmd_args
from gan4dag.data_generator import LsemDataset
import random
import numpy as np
from gan4dag.gan_model import GenNet, DiscNet, DiscGIN


D_Loss = torch.nn.BCEWithLogitsLoss(reduction='mean')


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

    if cmd_args.baseline:
        # ---------------------
        #  Observations -> DAGs (Run NOTEARS)
        # ---------------------
        db.train_to_dag()

    # ---------------------
    #  Initialize Networks
    # ---------------------

    print('*** Initializing networks ***')

    if cmd_args.baseline:
        hidden_dim = '32-32'
        output_hidden_dim = '64-1'
        act = output_act = 'relu'
    else:
        hidden_dim = cmd_args.f_hidden_dim
        output_hidden_dim = cmd_args.output_hidden_dim
        act = cmd_args.f_act
        output_act = cmd_args.output_act
    hp_arch = 'f-%s-%s-out-%s-%s' % (hidden_dim, act, output_hidden_dim, output_act)
    hp_train = 'm-%d-n-%d-gen-%d-bs-%d-glr-%.5f-dlr-%.5f' % (cmd_args.num_dag, cmd_args.num_sample, num_sample_gen,
                                                             cmd_args.batch_size, cmd_args.g_lr, cmd_args.d_lr)
    model_dump = hp_arch + '-' + hp_train + '.dump'

    if cmd_args.learn_noise:
        noise_mean, noise_sd = None, None
    else:
        noise_mean, noise_sd = db.noise_mean, db.noise_sd

    gen_net = GenNet(d=d,
                     noise_mean=noise_mean,
                     noise_sd=noise_sd).to(DEVICE)
    if cmd_args.baseline:
        disc_net = DiscGIN(d=d,
                           hidden_dims=hidden_dim,
                           nonlinearity=act,
                           output_hidden_dims=output_hidden_dim,
                           output_nonlinearity=output_act).to(DEVICE)
    else:
        disc_net = DiscNet(d=d,
                           f_hidden_dims=hidden_dim,
                           f_nonlinearity=act,
                           output_hidden_dims=output_hidden_dim,
                           output_nonlinearity=output_act).to(DEVICE)

    if cmd_args.phase == 'train':
        # ---------------------
        #  Optimizer
        # ---------------------

        g_opt = OPTIMIZER[cmd_args.g_optimizer](gen_net.parameters(),
                                                lr=cmd_args.g_lr,
                                                weight_decay=cmd_args.weight_decay)
        d_opt = OPTIMIZER[cmd_args.d_optimizer](disc_net.parameters(),
                                                lr=cmd_args.d_lr,
                                                weight_decay=cmd_args.weight_decay)

        # ---------------------
        #  Trainer
        # ---------------------
        trainer = LsemTrainer(gen_net, disc_net, g_opt, d_opt, db, num_sample_gen=num_sample_gen, save_dir=cmd_args.save_dir,
                              model_dump=model_dump, save_itr=cmd_args.save_itr)
        trainer.train(epochs=cmd_args.num_epochs, batch_size=cmd_args.batch_size, baseline=cmd_args.baseline)

    if cmd_args.phase == 'test':
        # ---------------------
        #  Eval
        # ---------------------
        evaluator = Eval(database=db, save_dir=cmd_args.save_dir, model_dump=model_dump, save_itr=cmd_args.save_itr, baseline=cmd_args.baseline)

        result = evaluator.eval(gen_net, m_small=128, m_large=2048,  verbose=True, bw=1.0)
        # print('mmd: ')
        # print(result['mmd'][1])
        print('ce: ')
        print(result['ce'][1])
        print('parameter: ')
        print(result['parameter'][1])
