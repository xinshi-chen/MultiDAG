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
from multidag.sergio_dataset import SergioDataset



def train(cmd_args, db, real_se, real_gn, group_size=1, group_start=0):
    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    # G_DAG
    assert group_size <= db.K
    hyperparameter = {'rho': cmd_args.rho, 'lambda': cmd_args.ld, 'c': cmd_args.c, 'gamma': cmd_args.gamma,
                      'eta': cmd_args.eta, 'mu': cmd_args.mu, 'dual_interval': cmd_args.dual_interval,
                      'alpha': cmd_args.alpha}
    hp = ''
    for key in hyperparameter:
        hp += key + '-' + f'{hyperparameter[key]}' + '_'
    hp = hp[:-1]
    models = []
    K_mask = np.arange(group_start, group_start+group_size)
    if cmd_args.real_dir:
        g_dag = G_DAG(num_dags=group_size, p=db.p).to(DEVICE)
    else:
        g_dag = G_DAG(num_dags=group_size, p=cmd_args.p).to(DEVICE)

    if cmd_args.ld > 1 and cmd_args.c > 1:
        g_dag.T = db.Perm

    # ---------------------
    #  Optimizer
    # ---------------------
    g_opt = OPTIMIZER[cmd_args.optimizer](g_dag.parameters(),
                                          lr=cmd_args.g_lr,
                                          weight_decay=cmd_args.weight_decay)
    # ---------------------
    #  Trainer
    # ---------------------
    trainer = Trainer(se=real_se, gn=real_gn, g_dag=g_dag, optimizer=g_opt, data_base=db,
                      K_mask=K_mask, hyperparameters=hyperparameter)

    models.append(trainer.train(epochs=cmd_args.num_epochs, start_epoch=cmd_args.start_epoch))

    # --------------------------
    #  save model and results
    # --------------------------
    model_save_root = './saved_models/' + db.hp + '/' + hp
    if not os.path.isdir(model_save_root):
        os.makedirs(model_save_root)
    name = f'multidag_group_size-{group_size}-{group_start}-{group_start+group_size}.pkl'
    model_save_dir = model_save_root + '/' + name
    with open(model_save_dir, 'wb') as handle:
        pickle.dump(models, handle)

    return


if __name__ == '__main__':
    if cmd_args.real_dir:
        db = SergioDataset(cmd_args.real_dir)
        print(f'*** solving {db.hp}_group_size-{cmd_args.group_size}-{cmd_args.group_start}-{cmd_args.group_start+cmd_args.group_size} ***')
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
        X = db.X[cmd_args.group_start:cmd_args.group_start + cmd_args.group_size].detach().numpy()
        G = db.G[cmd_args.group_start:cmd_args.group_start + cmd_args.group_size]
        real_se = np.square(X - X@G).sum(axis=-1).mean()
        real_gn = np.linalg.norm(G, axis=0).sum()
        print('real se: ', real_se)
        print('real group norm: ', real_gn)
        train(cmd_args, db, real_se, real_gn, group_size=cmd_args.group_size, group_start=cmd_args.group_start)
    else:
        # results = {10: [], 20: [], 40: [], 80: [], 160: [], 320: [], 640: []}
        if cmd_args.K == 32:
            group_size = [1] * 32 + [2] * 16 + [4] * 8 + [8] * 4 + [16] * 5
            group_start = list(range(32)) + list(range(0, 32, 2)) + list(range(0, 32, 4)) + list(range(0, 28, 4)) + list(
                range(0, 20, 4))
            nums = [0, 32, 48, 56, 63, 68]
        elif cmd_args.K == 64:
            group_size = [1] * 64 + [2] * 32 + [4] * 16 + [8] * 8 + [16] * 4 + [32] * 2
            group_start = list(range(64)) + list(range(0, 64, 2)) + list(range(0, 64, 4)) + list(range(0, 64, 8)) + list(
                range(0, 64, 16)) + list(range(0, 64, 32))
            nums = list(range(0, 64, 16)) + list(range(64, 96, 8)) + list(range(96, 112, 4)) + list(range(112, 120, 2)) + list(range(120, 124)) + list(range(124, 126)) + [126]
            print(len(nums))
        elif cmd_args.K == 256:
            group_size = [1] * 256 + [2] * 128 + [4] * 64 + [8] * 32 + [16] * 16 + [32] * 8
            group_start = list(range(256)) + list(range(0, 256, 2)) + list(range(0, 256, 4)) + list(range(0, 256, 8)) + \
                list(range(0, 256, 16)) + list(range(0, 256, 32))
            nums = list(range(0, 256, 64)) + list(range(256, 384, 32)) + list(range(384, 448, 16)) + \
                   list(range(448, 480, 8)) + list(range(480, 496, 4)) + list(range(496, 504, 4)) + [504]
        else:
            raise NotImplementedError

        for k in range(nums[cmd_args.group_idx], nums[cmd_args.group_idx+1]):
            cmd_args.group_start = group_start[k]
            cmd_args.group_size = group_size[k]
            db = Dataset(p=cmd_args.p,
                         n=cmd_args.n_sample,
                         K=cmd_args.K,
                         s0=cmd_args.s0,
                         s=cmd_args.s,
                         d=cmd_args.d,
                         w_range=(0.5, 2.0), verbose=True)
            print(f'*** solving {db.hp}_group_size-{cmd_args.group_size}-{cmd_args.group_start}-{cmd_args.group_start+cmd_args.group_size} ***')
            X = db.X[cmd_args.group_start:cmd_args.group_start + cmd_args.group_size].detach().numpy()
            G = db.G[cmd_args.group_start:cmd_args.group_start + cmd_args.group_size]
            real_se = np.square(X - X@G).sum(axis=-1).mean()
            real_gn = np.linalg.norm(G, axis=0).sum()
            print('real se: ', real_se)
            print('real group norm: ', real_gn)
            train(cmd_args, db, real_se, real_gn, group_size=cmd_args.group_size, group_start=cmd_args.group_start)
