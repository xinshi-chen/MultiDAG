import torch
from vae4dag.common.consts import DEVICE, OPTIMIZER
from vae4dag.trainer import Trainer
from vae4dag.eval import Eval
from vae4dag.common.cmd_args import cmd_args
from vae4dag.data_generator import Dataset
import random
import numpy as np
from vae4dag.model import Encoder, Decoder


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
    db = Dataset(d, W_sparsity=sparsity, W_threshold=threshold, num_dags=num_dag, num_sample=num_sample,
                 f_hidden_dims=cmd_args.true_f_hidden_dim, f_act=cmd_args.true_f_act, g_hidden_dims=None, g_act=None,
                 verbose=True)

    # ---------------------
    #  Initialize Networks
    # ---------------------

    print('*** Initializing networks ***')

    # Encoder

    mlp_dim = cmd_args.mlp_hidden_dim
    mlp_act = cmd_args.mlp_act
    tf_nhead = cmd_args.tf_nhead
    tf_num_stacks = cmd_args.tf_num_stacks
    tf_ff_dim = cmd_args.tf_ff_dim
    tf_act = cmd_args.tf_act
    temp = cmd_args.temperature

    encoder = Encoder(d, tf_nhead, tf_num_stacks, tf_ff_dim, 0.0, tf_act, mlp_dim, mlp_act, temp).to(DEVICE)

    # Decoder

    f_dims = cmd_args.f_hidden_dim
    f_act = cmd_args.f_act
    decoder = Decoder(d, f_hidden_dims=f_dims, f_act=f_act, learn_sd=False)

    hp_train = 'm-%d-n-%d-gen-%d-bs-%d-elr-%.5f-dlr-%.5f' % (cmd_args.num_dag, cmd_args.num_sample, num_sample_gen,
                                                             cmd_args.batch_size, cmd_args.e_lr, cmd_args.d_lr)
    hp_arch_enc = "-".join([mlp_dim, mlp_act, str(tf_nhead), str(tf_num_stacks), str(tf_ff_dim), tf_act, str(temp)])
    hp_arch_dec = "-".join([f_dims, f_act])

    model_dump = "-".join([hp_arch_enc, hp_arch_dec, hp_train]) + '.dump'

    if cmd_args.phase == 'train':
        # ---------------------
        #  Optimizer
        # ---------------------

        e_opt = OPTIMIZER[cmd_args.e_optimizer](encoder.parameters(),
                                                lr=cmd_args.e_lr,
                                                weight_decay=cmd_args.weight_decay)
        d_opt = OPTIMIZER[cmd_args.d_optimizer](decoder.parameters(),
                                                lr=cmd_args.d_lr,
                                                weight_decay=cmd_args.weight_decay)

        # ---------------------
        #  Trainer
        # ---------------------
        trainer = Trainer(encoder, decoder, e_opt, d_opt, db, num_sample_gen, save_dir=cmd_args.save_dir,
                          model_dump=model_dump, save_itr=cmd_args.save_itr, constraint_type=cmd_args.hw_type,
                          hyperparameters={'rho': cmd_args.rho,
                                           'gamma': cmd_args.gamma,
                                           'lambda': cmd_args.ld,
                                           'c': cmd_args.c,
                                           'eta': cmd_args.eta})

        trainer.train(epochs=cmd_args.num_epochs, batch_size=cmd_args.batch_size, start_epoch=cmd_args.start_epoch)

    # if cmd_args.phase == 'test':
    #     # ---------------------
    #     #  Eval
    #     # ---------------------
    #     evaluator = Eval(database=db, save_dir=cmd_args.save_dir, model_dump=model_dump, save_itr=cmd_args.save_itr, baseline=cmd_args.baseline)
    #
    #     result = evaluator.eval(gen_net, m_small=128, m_large=2048,  verbose=True, bw=1.0)
    #     # print('mmd: ')
    #     # print(result['mmd'][1])
    #     print('ce: ')
    #     print(result['ce'][1])
    #     print('parameter: ')
    #     print([x[0] for x in result['parameter'][1]])
