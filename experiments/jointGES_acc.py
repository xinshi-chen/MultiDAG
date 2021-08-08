import numpy as np
import pickle, os, glob
import matplotlib.pyplot as plt
from multidag.common.cmd_args import cmd_args
from multidag.data_generator import Dataset
from multidag.dag_utils import count_accuracy
from multidag.model import LSEM
from multidag.model import G_DAG
from torch import FloatTensor
from copy import deepcopy


p = [32, 64, 128, 256] #, 512, 1024]
n_samples = [10, 20, 40, 80, 160, 320]
s0 = [40, 96, 224, 512, 1152, 2560]
s = [120, 288, 672, 1536, 3456, 7680]
d = [5, 6, 7, 8, 9, 10]
sizes = [1, 2, 4, 8, 16, 32]

for idx in range(len(p)):
    # group_size * sample_size * task * configs
    title = ['fdr', 'tpr', 'fpr', 'shd', 'nnz']
    fdr = {key: {n: [] for n in n_samples} for key in sizes}
    tpr = {key: {n: [] for n in n_samples} for key in sizes}
    fpr = {key: {n: [] for n in n_samples} for key in sizes}
    shd = {key: {n: [] for n in n_samples} for key in sizes}
    nnz = {key: {n: [] for n in n_samples} for key in sizes}
    fdr_std = deepcopy(fdr)
    tpr_std = deepcopy(tpr)
    fpr_std = deepcopy(fpr)
    shd_std = deepcopy(shd)
    nnz_std = deepcopy(nnz)
    for n in n_samples:
        db = Dataset(p=p[idx],
                     n=n,
                     K=cmd_args.K,
                     s0=s0[idx],
                     s=s[idx],
                     d=d[idx],
                     w_range=(0.5, 2.0), verbose=False)
        # temp_x = db.X.detach().numpy()
        root = f'./saved_models/p-{p[idx]}_n-{n}_K-{cmd_args.K}_s-{s[idx]}_s0-{s0[idx]}_d-{d[idx]}_' \
               f'w_range_l-0.5_w_range_u-2.0/'
        for size in sizes:
            dir = os.path.join(root, f'jointGES-group_size-{size}.pkl')
            with open(dir, 'rb') as handle:
                t, A = pickle.load(handle)
            for k in range(cmd_args.K):
                G_true = np.abs(np.sign(db.G[k]))
                G_est = np.abs(A[k])
                G_est[G_est < 0.5] = 0
                G_est = np.sign(G_est)
                r = count_accuracy(G_true, G_est)
                fdr[size][n].append(r['fdr'])
                tpr[size][n].append(r['tpr'])
                fpr[size][n].append(r['fpr'])
                shd[size][n].append(r['shd'])
                nnz[size][n].append(r['nnz'])
    for size in sizes:
        for n in n_samples:
            temp_fdr = list(fdr[size][n])
            temp_tpr = list(tpr[size][n])
            temp_fpr = list(fpr[size][n])
            temp_shd = list(shd[size][n])
            temp_nnz = list(nnz[size][n])
            fdr[size][n] = np.mean(temp_fdr)
            tpr[size][n] = np.mean(temp_tpr)
            fpr[size][n] = np.mean(temp_fpr)
            shd[size][n] = np.mean(temp_shd)
            nnz[size][n] = np.mean(temp_nnz)
            fdr_std[size][n] = np.std(temp_fdr)
            tpr_std[size][n] = np.std(temp_tpr)
            fpr_std[size][n] = np.std(temp_fpr)
            shd_std[size][n] = np.std(temp_shd)
            nnz_std[size][n] = np.std(temp_nnz)
    fdr = np.array([[fdr[size][n] for n in n_samples] for size in sizes])
    tpr = np.array([[tpr[size][n] for n in n_samples] for size in sizes])
    fpr = np.array([[fpr[size][n] for n in n_samples] for size in sizes])
    shd = np.array([[shd[size][n] for n in n_samples] for size in sizes])
    nnz = np.array([[nnz[size][n] for n in n_samples] for size in sizes])

    fdr_std = np.array([[fdr_std[size][n] for n in n_samples] for size in sizes])
    tpr_std = np.array([[tpr_std[size][n] for n in n_samples] for size in sizes])
    fpr_std = np.array([[fpr_std[size][n] for n in n_samples] for size in sizes])
    shd_std = np.array([[shd_std[size][n] for n in n_samples] for size in sizes])
    nnz_std = np.array([[nnz_std[size][n] for n in n_samples] for size in sizes])
    # for k in range(len(sizes)):
    #     ax[0, idx].plot(n_samples, fdr[k], color=color[k], label=label[k])
    #     ax[1, idx].plot(n_samples, tpr[k], color=color[k], label=label[k])
    #     ax[2, idx].plot(n_samples, fpr[k], color=color[k], label=label[k])
    #     ax[3, idx].plot(n_samples, shd[k], color=color[k], label=label[k])
    #     ax[4, idx].plot(n_samples, nnz[k], color=color[k], label=label[k])
    #     ax[0, idx].fill_between(n_samples, fdr[k] - fdr_std[k], fdr[k] + fdr_std[k], alpha=0.3, color=color[k])
    #     ax[1, idx].fill_between(n_samples, tpr[k] - tpr_std[k], tpr[k] + tpr_std[k], alpha=0.3, color=color[k])
    #     ax[2, idx].fill_between(n_samples, fpr[k] - fpr_std[k], fpr[k] + fpr_std[k], alpha=0.3, color=color[k])
    #     ax[3, idx].fill_between(n_samples, shd[k] - shd_std[k], shd[k] + shd_std[k], alpha=0.3, color=color[k])
    #     ax[4, idx].fill_between(n_samples, nnz[k] - nnz_std[k], nnz[k] + nnz_std[k], alpha=0.3, color=color[k])
        # ax[0, idx].errorbar(n_samples, fdr[k], yerr=fdr_std[k], color=color[k], label=label[k])
        # ax[1, idx].errorbar(n_samples, tpr[k], yerr=tpr_std[k], color=color[k], label=label[k])
        # ax[2, idx].errorbar(n_samples, fpr[k], yerr=fpr_std[k], color=color[k], label=label[k])
        # ax[3, idx].errorbar(n_samples, shd[k], yerr=shd_std[k], color=color[k], label=label[k])
        # ax[4, idx].errorbar(n_samples, nnz[k], yerr=nnz_std[k], color=color[k], label=label[k])
    # ax[0, idx].set_title(f'p = {p[idx]}', fontsize=16)
    # for k in range(5):
    #     ax[k, idx].legend(prop={'size': 15})
    #     ax[k, idx].set_xscale('log', base=2)
    #     ax[k, idx].grid(axis='y', linestyle='dashed')
    # ax[-1, idx].set_xlabel('n (number of samples)')
    # for item in [ax[-1,idx].xaxis.label] + [ax[k,idx].yaxis.label for k in range(5)] +\
    #             ax[0,idx].get_xticklabels() + ax[0,idx].get_yticklabels() + ax[1,idx].get_xticklabels() +\
    #             ax[1,idx].get_yticklabels() + ax[2,idx].get_xticklabels() + ax[2,idx].get_yticklabels() +\
    #             ax[3,idx].get_xticklabels() + ax[3,idx].get_yticklabels() + ax[4,idx].get_xticklabels() +\
    #             ax[4,idx].get_yticklabels():
    #     item.set_fontsize(16)
    # print statistics
    print(f'### fdr for p = {p[idx]} ###')
    for k in range(len(sizes)):
        log = f'k={k} & '
        for l in range(len(n_samples)):
            log += f'${fdr[k][l]:.4f}\pm{fdr_std[k][l]:.4f}$ & '
        print(log[:-2] + '\\')
    print(f'### tpr for p = {p[idx]} ###')
    for k in range(len(sizes)):
        log = f'k={k} & '
        for l in range(len(n_samples)):
            log += f'${tpr[k][l]:.4f}\pm{tpr_std[k][l]:.4f}$ & '
        print(log[:-2] + '\\')
    print(f'### fpr for p = {p[idx]} ###')
    for k in range(len(sizes)):
        log = f'k={k} & '
        for l in range(len(n_samples)):
            log += f'${fpr[k][l]:.4f}\pm{fpr_std[k][l]:.4f}$ & '
        print(log[:-2] + '\\')
    print(f'### shd for p = {p[idx]} ###')
    for k in range(len(sizes)):
        log = f'k={k} & '
        for l in range(len(n_samples)):
            log += f'${shd[k][l]:.4f}\pm{shd_std[k][l]:.4f}$ & '
        print(log[:-2] + '\\')
    print(f'### nnz for p = {p[idx]} ###')
    for k in range(len(sizes)):
        log = f'k={k} & '
        for l in range(len(n_samples)):
            log += f'${nnz[k][l]:.4f}\pm{nnz_std[k][l]:.4f}$ & '
        print(log[:-2] + '\\')

# ax[0, 0].set_ylabel('False Discovery Rate (FDR)')
# ax[1, 0].set_ylabel('True Positive Rate (TPR)')
# ax[2, 0].set_ylabel('False Positive Rate (FPR)')
# ax[3, 0].set_ylabel('Structure Hamming Distance (SHD)')
# ax[4, 0].set_ylabel('Number of Non-Zeros (NNZ)')
# # plt.show()
# plt.savefig(f'figs/acc.pdf', bbox_inches='tight')

