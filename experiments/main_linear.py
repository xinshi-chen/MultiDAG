import torch
import os, pickle
from multidag.common.consts import DEVICE, OPTIMIZER
from multidag.trainer import Trainer
from multidag.common.cmd_args import cmd_args
from multidag.data_generator import Dataset
import random
import numpy as np
import time
from multidag.model import G_DAG



def train_and_evaluate(cmd_args, db, group_size=1):
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    # G_DAG
    assert group_size <= db.K
    result = []
    state_dict = []
    for i in range(db.K // group_size):
        g_dag = G_DAG(num_dags=group_size, p=cmd_args.p).to(DEVICE)

        # ---------------------
        #  Optimizer
        # ---------------------
        g_opt = OPTIMIZER[cmd_args.optimizer](g_dag.parameters(),
                                              lr=cmd_args.g_lr,
                                              weight_decay=cmd_args.weight_decay)
        # ---------------------
        #  Trainer
        # ---------------------
        trainer = Trainer(g_dag=g_dag, optimizer=g_opt, data_base=db,
                          K_mask=np.arange(i*group_size, (i+1)*group_size))

        state_dict.append(trainer.train(epochs=cmd_args.num_epochs + 1000 * int(np.log2(cmd_args.n / 10)), start_epoch=cmd_args.start_epoch))

        # ---------------------
        #  Eval
        # ---------------------
        result.extend(trainer.evaluate())

    # --------------------------
    #  save model and results
    # --------------------------
    model_save_root = './saved_models/' + db.hp
    result_save_root = './results/' + db.hp
    if not os.path.isdir(model_save_root):
        os.makedirs(model_save_root)
    if not os.path.isdir(result_save_root):
        os.makedirs(result_save_root)
    name = f'multidag_group_size-{group_size}.pkl'
    model_save_dir = model_save_root + '/' + name
    result_save_dir = result_save_root + '/' + name
    with open(model_save_dir, 'wb') as handle:
        pickle.dump(state_dict, handle)
    with open(result_save_dir, 'wb') as handle:
        pickle.dump(result, handle)

    return result


if __name__ == '__main__':
    # check K is the power of 2
    assert (cmd_args.K & (cmd_args.K - 1) == 0) and cmd_args.K != 0
    db = Dataset(p=cmd_args.p,
                 n=cmd_args.n,
                 K=cmd_args.K,
                 s0=cmd_args.s0,
                 s=cmd_args.s,
                 d=cmd_args.d,
                 w_range=(0.5, 2.0), verbose=True)
    group_size = 1
    while group_size <= cmd_args.K:
        print(f'*** solving {db.hp}_group_size-{group_size} ***')
        result = train_and_evaluate(cmd_args, db, group_size=group_size)
        group_size *= 2