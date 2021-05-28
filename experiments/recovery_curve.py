import numpy as np
import pickle, os, glob
import matplotlib.pyplot as plt
import seaborn as sns
from multidag.data_generator import Dataset
from multidag.common.cmd_args import cmd_args
from multidag.dag_utils import is_dag, project_to_dag
from multidag.model import G_DAG

def success(B_est, perm, threshold=1, verbose=False):
    '''
    check whether the topo order is successfully recovered
    '''
    if not is_dag(B_est):
        if verbose:
            print('B_est is not DAG, use projection')
        B_est, _ = project_to_dag(B_est)
    inv = np.linalg.inv(perm)
    target = inv.T.dot(B_est).dot(inv)
    if np.abs(np.triu(target, 1)).sum() > 1e-6:
        return 0
    elif np.abs(np.tril(target, 1)).sum() < threshold:
        return 0
    else:
        return 1

def accuracy(result):
    accs = []
    m, n = result.shape
    target = np.flip(result, axis=1)
    for k in range(n-1, -m, -1):
        accs.append(np.mean(target.diagonal(offset=k)))
    return accs

p = [32, 64, 128, 256] #, 512, 1024]
n_samples = [10, 20, 40, 80, 160, 320]
s0 = [40, 96, 224, 512, 1152, 2560]
s = [120, 288, 672, 1536, 3456, 7680]
d = [5, 6, 7, 8, 9, 10]
sizes = [1, 2, 4, 8, 16, 32]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12,9))
color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
label = [f'p={pp}' for pp in p]
marker = ['.', 'P', 'o', 'v']
linestyle = ['-', '--', '-.', ':', '']
for idx in range(len(p)):
    # group_size * sample_size * task * configs
    result = {key: {n: [[] for _ in range(cmd_args.K)] for n in n_samples} for i, key in enumerate(sizes)}
    for n in n_samples:
        db = Dataset(p=p[idx],
                     n=n,
                     K=cmd_args.K,
                     s0=s0[idx],
                     s=s[idx],
                     d=d[idx],
                     w_range=(0.5, 2.0), verbose=False)
        perm = db.Perm
        root = f'./saved_models/p-{p[idx]}_n-{n}_K-{cmd_args.K}_s-{s[idx]}_s0-{s0[idx]}_d-{d[idx]}_' \
               f'w_range_l-0.5_w_range_u-2.0'
        dir_list = os.listdir(root)
        # each dir represents a hyperparameter configuration
        for dir in dir_list:
            file_list = glob.glob(os.path.join(root, dir) + '/*.pkl')
            # each file represent a solution for corresponding tasks
            for file in file_list:
                with open(file, 'rb') as handle:
                    model = pickle.load(handle)[0]
                basename = os.path.splitext(os.path.basename(file))[0]
                s_e = basename.split('-')[-2:]
                size = int(s_e[1]) - int(s_e[0])
                G_est = np.abs(model['G'] * model['T'])
                G_est[G_est < cmd_args.threshold + 0.02 * np.log2(n / 10)] = 0
                # G_est[G_est < cmd_args.threshold] = 0
                G_est = np.sign(G_est)
                for m in range(G_est.shape[0]):
                    r = success(G_est[m], perm, threshold=p[idx]/3)
                    result[size][n][int(s_e[0]) + m].append(r)
    for size in sizes:
        for n in n_samples:
            temp = list(result[size][n])
            result[size][n] = np.mean(temp)
    result = np.array([[result[size][n] for size in sizes] for n in n_samples])
    print(result)
    # ax[idx].imshow(result, cmap='hot', interpolation='nearest')
    y = accuracy(result)
    x = [np.sqrt(10 * 2**i * p[idx]/ (s0[idx]**2 * np.log(p[idx]))) for i in range(len(y))]
    ax.plot(x, y, color=color[idx], label=label[idx], linestyle=linestyle[idx], marker=marker[idx])
# ax[-1].set_xlim([0,2])
ax.legend()
ax.set_ylabel('Recovery Probability', fontsize=14)
ax.set_xlabel('1 / theta?', fontsize=14)
# plt.show()
plt.savefig(f'figs/curve.pdf', bbox_inches='tight')
