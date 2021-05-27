import numpy as np
import pickle, os, glob
import matplotlib.pyplot as plt
from multidag.common.cmd_args import cmd_args
from multidag.data_generator import Dataset
from multidag.dag_utils import count_accuracy
from multidag.model import G_DAG


title = ['fdr', 'tpr', 'fpr', 'shd', 'nnz']
p = [32, 64, 128]#, 256] #, 512, 1024]
n_samples = [10, 20, 40, 80, 160, 320, 640]
s0 = [40, 96, 224, 512, 1152, 2560]
s = [120, 288, 672, 1536, 3456, 7680]
d = [5, 6, 7, 8, 9, 10]
sizes = [1, 2, 4, 8, 16, 32, 64]

fig, ax = plt.subplots(nrows=len(title), ncols=len(p), figsize=(25,25))
color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
label = [f'k={k}' for k in sizes]
for idx in range(len(p)):
    # group_size * sample_size * task * configs
    title = ['fdr', 'tpr', 'fpr', 'shd', 'nnz']
    fdr = {key: {n: [[] for _ in range(cmd_args.K)] for n in n_samples} for i, key in enumerate(sizes)}
    tpr = {key: {n: [[] for _ in range(cmd_args.K)] for n in n_samples} for i, key in enumerate(sizes)}
    fpr = {key: {n: [[] for _ in range(cmd_args.K)] for n in n_samples} for i, key in enumerate(sizes)}
    shd = {key: {n: [[] for _ in range(cmd_args.K)] for n in n_samples} for i, key in enumerate(sizes)}
    nnz = {key: {n: [[] for _ in range(cmd_args.K)] for n in n_samples} for i, key in enumerate(sizes)}
    for n in n_samples:
        db = Dataset(p=p[idx],
                     n=n,
                     K=cmd_args.K,
                     s0=s0[idx],
                     s=s[idx],
                     d=d[idx],
                     w_range=(0.5, 2.0), verbose=False)
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
                K_mask = np.arange(int(s_e[0]), int(s_e[1]))
                G_true = np.abs(np.sign(db.G[K_mask]))
                G_est = np.abs(model['G'] * model['T'])
                # G_est[G_est < cmd_args.threshold + 0.02 * np.log2(n / 10)] = 0
                # G_est[G_est < cmd_args.threshold - 0.02 * np.log2(G_true.shape[0])] = 0
                G_est[G_est < cmd_args.threshold] = 0
                G_est = np.sign(G_est)
                for m in range(G_true.shape[0]):
                    r = count_accuracy(G_true[m], G_est[m])
                    fdr[size][n][int(s_e[0]) + m].append(r['fdr'])
                    tpr[size][n][int(s_e[0]) + m].append(r['tpr'])
                    fpr[size][n][int(s_e[0]) + m].append(r['fpr'])
                    shd[size][n][int(s_e[0]) + m].append(r['shd'])
                    nnz[size][n][int(s_e[0]) + m].append(r['nnz'])
    for size in sizes:
        for n in n_samples:
            temp_fdr = list(fdr[size][n])
            temp_tpr = list(tpr[size][n])
            temp_fpr = list(fpr[size][n])
            temp_shd = list(shd[size][n])
            temp_nnz = list(nnz[size][n])
            for k in range(cmd_args.K):
                temp_fdr[k] = np.mean(temp_fdr[k])
                temp_tpr[k] = np.mean(temp_tpr[k])
                temp_fpr[k] = np.mean(temp_fpr[k])
                temp_shd[k] = np.mean(temp_shd[k])
                temp_nnz[k] = np.mean(temp_nnz[k])
            fdr[size][n] = np.mean(temp_fdr)
            tpr[size][n] = np.mean(temp_tpr)
            fpr[size][n] = np.mean(temp_fpr)
            shd[size][n] = np.mean(temp_shd)
            nnz[size][n] = np.mean(temp_nnz)
    fdr = np.array([[fdr[size][n] for n in n_samples] for size in sizes])
    tpr = np.array([[tpr[size][n] for n in n_samples] for size in sizes])
    fpr = np.array([[fpr[size][n] for n in n_samples] for size in sizes])
    shd = np.array([[shd[size][n] for n in n_samples] for size in sizes])
    nnz = np.array([[nnz[size][n] for n in n_samples] for size in sizes])
    for k in range(len(sizes)):
        ax[0, idx].plot(n_samples, fdr[k], color=color[k], label=label[k])
        ax[1, idx].plot(n_samples, tpr[k], color=color[k], label=label[k])
        ax[2, idx].plot(n_samples, fpr[k], color=color[k], label=label[k])
        ax[3, idx].plot(n_samples, shd[k], color=color[k], label=label[k])
        ax[4, idx].plot(n_samples, nnz[k], color=color[k], label=label[k])
    ax[0, idx].title.set_text(f'p = {p[idx]}')
    for tt in range(5):
        ax[tt, idx].legend()
        ax[tt, idx].set_xscale('log', base=2)
    ax[4, idx].set_xlabel('n (number of samples)')
ax[0, 0].set_ylabel('False Discovery Rate (FDR)')
ax[1, 0].set_ylabel('True Positive Rate (TPR)')
ax[2, 0].set_ylabel('False Positive Rate (FPR)')
ax[3, 0].set_ylabel('Structure Hamming Distance (SHD)')
ax[4, 0].set_ylabel('Number of Non-Zeros (NNZ)')
# plt.show()
plt.savefig(f'figs/acc.pdf', bbox_inches='tight')