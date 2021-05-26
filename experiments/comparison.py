import os
import numpy as np
import pickle
from multidag.common.cmd_args import cmd_args
from multidag.data_generator import Dataset
from multidag.dag_utils import count_accuracy
from multidag.model import G_DAG
from multidag.sergio_dataset import SergioDataset
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

title = ['fdr', 'tpr', 'fpr', 'shd', 'nnz']
# notears_result = []
# with open(f'./results/p-{cmd_args.p}_n-{cmd_args.n}_K-{cmd_args.K}_s-{cmd_args.s}_s0-{cmd_args.s0}_'
#           f'd-{cmd_args.d}_w_range_l-0.5_w_range_u-2.0/notears.pkl', 'rb') as handle:
#     result = pickle.load(handle)
#     for r in result:
#         notears_result.append([r[key] for key in r])
# mean_notear_result = np.array(notears_result).mean(axis=0)
# print('###### notears average results #####')
# for i, t in enumerate(title):
#     print(f'{t}: {mean_notear_result[i]:.6f}')

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
group_size = cmd_args.group_start
while group_size <= cmd_args.K:
    g_dag = G_DAG(num_dags=group_size, p=cmd_args.p)
    multidag_result = []
    for i in range(db.K // group_size):
        if cmd_args.real_dir:
            with open(f'./saved_models/{cmd_args.real_dir}/rho-{cmd_args.rho}_lambda-{cmd_args.ld}_'
                      f'c-{cmd_args.c}_gamma-{cmd_args.gamma}_'
                      f'eta-{cmd_args.eta}_mu-{cmd_args.mu}_dual_interval-{cmd_args.dual_interval}/multidag_'
                      f'group_size-{group_size}-{i * group_size}-{(i+1) * group_size}.pkl', 'rb') as handle:
                model = pickle.load(handle)[0]
        else:
            with open(f'./saved_models/p-{cmd_args.p}_n-{cmd_args.n_sample}_K-{cmd_args.K}_s-{cmd_args.s}_s0-{cmd_args.s0}_'
                      f'd-{cmd_args.d}_w_range_l-0.5_w_range_u-2.0/rho-{cmd_args.rho}_lambda-{cmd_args.ld}_'
                      f'c-{cmd_args.c}_gamma-{cmd_args.gamma}_'
                      f'eta-{cmd_args.eta}_mu-{cmd_args.mu}_dual_interval-{cmd_args.dual_interval}/multidag_'
                      f'group_size-{group_size}-{i * group_size}-{(i+1) * group_size}.pkl', 'rb') as handle:
                model = pickle.load(handle)[0]
        K_mask = np.arange(i * group_size, (i + 1) * group_size)
        G_true = np.abs(np.sign(db.G[K_mask]))
        G_est = np.abs(model['G'] * model['T'])
        G_est[G_est < cmd_args.threshold] = 0
        G_est = np.sign(G_est)
        for k in range(G_true.shape[0]):
            r = count_accuracy(G_true[k], G_est[k])
            multidag_result.append([r[key] for key in r])
        fig, axs = plt.subplots(1, 4)
        axs[0].imshow(G_true[0])
        axs[0].set_title('True')
        axs[1].imshow(G_est[0])
        axs[1].set_title('Recovered (G*T)')
        axs[2].imshow(np.abs(model['T'])[0])
        axs[2].set_title('T')
        axs[3].imshow(np.abs(model['G'])[0])
        axs[3].set_title('G')
        plt.show()
    mean_multidag_result = np.array(multidag_result).mean(axis=0)
    print(f'###### multidag average results for group_size {group_size} #####')
    for i, t in enumerate(title):
        print(f'{t}: {mean_multidag_result[i]:.6f}')
    group_size *= 2

