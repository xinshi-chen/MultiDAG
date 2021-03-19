import torch
import math
from vae4dag.common.consts import DEVICE, OPTIMIZER
from vae4dag.trainer import Trainer
from vae4dag.eval import Eval, eval_structure
from vae4dag.common.cmd_args import cmd_args
from vae4dag.data_generator import GenDataset
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

    num_sample = cmd_args.num_sample
    threshold = cmd_args.threshold
    sparsity = cmd_args.sparsity
    d = cmd_args.d

    # ---------------------
    #  Synthetic Dataset
    # ---------------------

    print('*** Loading data ***')
    db = GenDataset(d, num_sample, W_sparsity=sparsity, W_threshold=threshold, f_hidden_dims=cmd_args.true_f_hidden_dim,
                    f_act=cmd_args.true_f_act, g_hidden_dims=None, g_act=None, verbose=True, num_test=None,
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

    encoder = Encoder(d, tf_nhead, tf_num_stacks, tf_ff_dim, 0.0, tf_act, mlp_dim, mlp_act, temp).to(DEVICE)

    # Decoder

    f_dims = cmd_args.f_hidden_dim
    f_act = cmd_args.f_act
    decoder = Decoder(d, f_hidden_dims=f_dims, f_act=f_act, learn_sd=False).to(DEVICE)

    hp_train = 'bs-%d-elr-%.5f-dlr-%.5f' % (cmd_args.batch_size, cmd_args.e_lr, cmd_args.d_lr)
    hp_arch_enc = "-".join([mlp_dim, mlp_act, str(tf_nhead), str(tf_num_stacks), str(tf_ff_dim), tf_act, str(temp)])
    hp_arch_dec = "-".join([f_dims, f_act])

    model_dump = "-".join([hp_arch_enc, hp_arch_dec, hp_train]) + '.dump'

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
    X, _ = db.static_data['train']
    k = math.floor(db.n * cmd_args.p)
    X = X[:, :k, :]
    W = db.static_dag['train']

    Trainer.train_encoder_with_W(encoder, e_opt, X, W, epochs=1000, batch_size=cmd_args.batch_size)
    # ---------------------
    #  Eval
    # ---------------------
    with torch.no_grad():
        X, _ = db.static_data['test']
        X = X[:, :k, :]
        W = db.static_dag['test']
        W_est = encoder(X.to(DEVICE))
        print(W_est)
        W_est = Eval.project_W(W_est, device=DEVICE, verbose=True)
        print(W)
        print(W_est)
    # compare structure

    result = eval_structure(W_est, W)
    print(result)
    for key in result:
        print(key)
        print(np.array(result[key]).mean())
