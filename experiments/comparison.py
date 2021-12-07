import numpy as np
import pickle
from multidag.common.cmd_args import cmd_args
from multidag.data_generator import Dataset
from multidag.dag_utils import count_accuracy
from multidag.model import G_DAG
from multidag.sergio_dataset import SergioDataset

title = ['fdr', 'tpr', 'fpr', 'shd', 'nnz']


if cmd_args.real_dir:
    db = SergioDataset(f'../data/{cmd_args.real_dir}')
else:
    db = Dataset(p=cmd_args.p,
                 n=cmd_args.n_sample,
                 K=cmd_args.K,
                 s0=cmd_args.s0,
                 s=cmd_args.s,
                 d=cmd_args.d,
                 w_range=(0.5, 2.0), verbose=True)

group_size = [1] * 32 + [2] * 16 + [4] * 8 + [8] * 7 + [16] * 5
group_start = list(range(32)) + list(range(0, 32, 2)) + list(range(0, 32, 4)) + list(range(0, 28, 4)) + list(range(0, 20, 4))
nums = [0, 32, 48, 56, 63, 68]

for i in range(len(nums) - 1):
    multidag_result = []
    for j in range(nums[i], nums[i+1]):
        with open(f'./saved_models/p-{cmd_args.p}_n-{cmd_args.n_sample}_K-{cmd_args.K}_s-{cmd_args.s}_s0-{cmd_args.s0}_'
                  f'd-{cmd_args.d}_w_range_l-0.5_w_range_u-2.0/rho-{cmd_args.rho}_lambda-{cmd_args.ld}_'
                  f'c-{cmd_args.c}_gamma-{cmd_args.gamma}_'
                  f'eta-{cmd_args.eta}_mu-{cmd_args.mu}_dual_interval-{cmd_args.dual_interval}/multidag_'
                  f'group_size-{group_size[j]}-{group_start[j]}-{group_start[j] + group_size[j]}.pkl', 'rb') as handle:
            model = pickle.load(handle)[0]
        K_mask = np.arange(group_start[j], group_start[j] + group_size[j])
        G_true = np.abs(np.sign(db.G[K_mask]))
        G_est = np.abs(model['G'] * model['T'])
        G_est[G_est < cmd_args.threshold] = 0
        G_est = np.sign(G_est)
        for k in range(G_true.shape[0]):
            r = count_accuracy(G_true[k], G_est[k])
            multidag_result.append([r[key] for key in r])
    mean_multidag_result = np.array(multidag_result).mean(axis=0)
    print(f'###### multidag average results for group_size {group_size[nums[i]]} #####')
    for k, t in enumerate(title):
        print(f'{t}: {mean_multidag_result[k]:.6f}')

