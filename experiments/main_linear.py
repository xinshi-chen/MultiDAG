import torch
from multidag.common.consts import DEVICE, OPTIMIZER
from multidag.trainer import Trainer
from multidag.common.cmd_args import cmd_args
from multidag.data_generator import Dataset
import random
import numpy as np
from multidag.model import G_DAG


if __name__ == '__main__':

    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    # ---------------------
    #  Synthetic Dataset
    # ---------------------

    print('*** Loading data ***')
    db = Dataset(p=cmd_args.p,
                 n=cmd_args.n,
                 K=cmd_args.K,
                 s0=cmd_args.s0,
                 s=cmd_args.s,
                 d=cmd_args.d,
                 gid=cmd_args.gid,
                 xid=cmd_args.xid,
                 w_range=(0.5, 2.0), verbose=True)

    # ---------------------
    #  Initialize Networks
    # ---------------------

    print('*** Initializing networks ***')

    # Encoder

    # G_DAG
    g_dag = G_DAG(num_dags=cmd_args.K, p=cmd_args.p).to(DEVICE)

    # ---------------------
    #  Optimizer
    # ---------------------
    g_opt = OPTIMIZER[cmd_args.optimizer](g_dag.parameters(),
                                          lr=cmd_args.g_lr,
                                          weight_decay=cmd_args.weight_decay)
    # ---------------------
    #  Trainer
    # ---------------------
    trainer = Trainer(g_dag=g_dag, optimizer=g_opt, data_base=db, save_dir='./saved_models',
                      model_dump=f'parameters')
    if cmd_args.phase == 'train':
        trainer.train(epochs=cmd_args.num_epochs, start_epoch=cmd_args.start_epoch)

    # ---------------------
    #  Eval
    # ---------------------
    print('*** Evaluation ***')
    trainer.evaluate()
    # load model
    # evaluate
    # compare structure
