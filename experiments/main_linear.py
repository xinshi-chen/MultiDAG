import torch
from vae4dag.common.consts import DEVICE, OPTIMIZER
from vae4dag.trainer import Trainer
from vae4dag.eval import Eval, eval_structure
from vae4dag.common.cmd_args import cmd_args
from vae4dag.data_generator import GenDataset
import random
import numpy as np
from vae4dag.model import Encoder, Decoder, DecoderLinear, W_DAG


if __name__ == '__main__':

    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    # ---------------------
    #  Hyperparameters
    # ---------------------

    num_sample = cmd_args.num_sample
    threshold = cmd_args.threshold
    sparsity = cmd_args.sparsity
    d = cmd_args.d

    # ---------------------
    #  Synthetic Dataset
    # ---------------------

    print('*** Loading data ***')
    db = GenDataset(d, num_sample, W_sparsity=sparsity, W_threshold=threshold, f_hidden_dims=None, f_act=None,
                    g_hidden_dims=None, g_act=None, verbose=True, num_test=None,
                    num_dags={'train': cmd_args.num_train,
                              'vali': cmd_args.num_vali,
                              'test': cmd_args.num_test})

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

    encoder = Encoder(d, tf_nhead, tf_num_stacks, tf_ff_dim, 0.0, tf_act, mlp_dim, mlp_act).to(DEVICE)

    # W_DAG
    w_dag = W_DAG(db.num_dags['train'], d).to(DEVICE)

    # Decoder
    decoder = DecoderLinear()

    # f_dims = cmd_args.f_hidden_dim
    # f_act = cmd_args.f_act
    # decoder = Decoder(d, f_hidden_dims=f_dims, f_act=f_act, learn_sd=False).to(DEVICE)

    hp_train = [cmd_args.batch_size, cmd_args.e_lr, cmd_args.d_lr, cmd_args.w_lr]
    hp_train = "-".join(list(map(str, hp_train)))
    hp_arch_enc = "-".join([mlp_dim, mlp_act, str(tf_nhead), str(tf_num_stacks), str(tf_ff_dim), tf_act, str(temp)])
    hp_arch_dec = ""

    model_dump = "-".join([hp_arch_enc, hp_arch_dec, hp_train]) + '.dump'

    # ---------------------
    #  Optimizer
    # ---------------------

    e_opt = OPTIMIZER[cmd_args.e_optimizer](encoder.parameters(),
                                            lr=cmd_args.e_lr,
                                            weight_decay=cmd_args.weight_decay)
    # d_opt = OPTIMIZER[cmd_args.d_optimizer](decoder.parameters(),
    #                                         lr=cmd_args.d_lr,
    #                                         weight_decay=cmd_args.weight_decay)
    d_opt = None

    w_opt = OPTIMIZER[cmd_args.w_optimizer](w_dag.parameters(),
                                            lr=cmd_args.w_lr,
                                            weight_decay=cmd_args.weight_decay)

    # ---------------------
    #  Trainer
    # ---------------------
    trainer = Trainer(encoder, decoder, w_dag, e_opt, d_opt, w_opt, db, save_dir=cmd_args.save_dir,
                      model_dump=model_dump, save_itr=cmd_args.save_itr, constraint_type=cmd_args.hw_type,
                      hyperparameters={'rho': cmd_args.rho,
                                       'alpha': cmd_args.alpha,
                                       'lambda': cmd_args.ld,
                                       'c': cmd_args.c,
                                       'p': cmd_args.p,
                                       'eta': cmd_args.eta})
    if cmd_args.phase == 'train':
        trainer.train(epochs=cmd_args.num_epochs, batch_size=cmd_args.batch_size, start_epoch=cmd_args.start_epoch)
        print(w_dag.w[0:4])

    # ---------------------
    #  Eval
    # ---------------------

    # load model
    dump = trainer.save_dir + '/best-' + trainer.model_dump
    encoder.load_state_dict(torch.load(dump))
    if d_opt is not None:
        dump = dump[:-5] + '_decoder.dump'
        decoder.load_state_dict(torch.load(dump))
    dump = dump[:-5] + '_wD.dump'
    w_dag.load_state_dict(torch.load(dump))
    # evaluate
    true_nll_in, nll_in, true_nll_eval, nll_eval, W_est = Eval.eval(encoder, decoder, db, phase='test', k=trainer.k,
                                                                    verbose=True, return_W=True)
    # compare structure
    W_true = db.static_dag['test']
    result = eval_structure(W_est, W_true)
    print(result)
    for key in result:
        print(key)
        print(np.array(result[key]).mean())

    print(trainer.model_dump)
    print(trainer.save_dir)
