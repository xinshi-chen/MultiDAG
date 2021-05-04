import numpy as np
import pickle
from multidag.common.cmd_args import cmd_args

title = ['fdr', 'tpr', 'fpr', 'shd', 'nnz']
notears_result = []
for gid in range(cmd_args.num_g):
    for xid in range(cmd_args.num_x):
        with open(f'./results/p-{cmd_args.p}_n-{cmd_args.n}_K-{cmd_args.K}_s-{cmd_args.s}_s0-{cmd_args.s0}_'
                  f'd-{cmd_args.d}_w_range_l-0.5_w_range_u-2.0/notears_'
                  f'gid-{gid}_xid-{xid}.pkl', 'rb') as handle:
            result = pickle.load(handle)
            for r in result:
                notears_result.append([r[key] for key in r])
mean_notear_result = np.array(notears_result).mean(axis=0)
print('###### notears average results #####')
for i, t in enumerate(title):
    print(f'{t}: {mean_notear_result[i]:.6f}')

batch_size = 1
while batch_size <= cmd_args.K:
    multidag_result = []
    for gid in range(cmd_args.num_g):
        for xid in range(cmd_args.num_x):
            with open(f'./results/p-{cmd_args.p}_n-{cmd_args.n}_K-{cmd_args.K}_s-{cmd_args.s}_s0-{cmd_args.s0}_'
                      f'd-{cmd_args.d}_w_range_l-0.5_w_range_u-2.0/multidag_'
                      f'gid-{gid}_xid-{xid}_batch_size-{batch_size}.pkl', 'rb') as handle:
                result = pickle.load(handle)
                for r in result:
                    multidag_result.append([r[key] for key in r])
    mean_multidag_result = np.array(multidag_result).mean(axis=0)
    print(f'###### notears average results for batch_size {batch_size} #####')
    for i, t in enumerate(title):
        print(f'{t}: {mean_multidag_result[i]:.6f}')
    batch_size *= 2

