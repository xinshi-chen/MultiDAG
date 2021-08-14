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


p = [32]#, 64, 128, 256] #, 512, 1024]
n_samples = [10, 20, 40, 80, 160, 320]
s0 = [40, 96, 224, 512, 1152, 2560]
s = [120, 288, 672, 1536, 3456, 7680]
d = [5, 6, 7, 8, 9, 10]
sizes = [1]#, 2, 4, 8, 16, 32]

for idx in range(len(p)):
    # group_size * sample_size * task * configs
    title = ['fdr', 'tpr', 'fpr', 'shd', 'nnz']
    t = {key: {n: [] for n in n_samples} for key in sizes}
    fdr = {key: {n: [] for n in n_samples} for key in sizes}
    tpr = {key: {n: [] for n in n_samples} for key in sizes}
    fpr = {key: {n: [] for n in n_samples} for key in sizes}
    shd = {key: {n: [] for n in n_samples} for key in sizes}
    nnz = {key: {n: [] for n in n_samples} for key in sizes}
    t_std = {key: {n: [] for n in n_samples} for key in sizes}
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
        root = f'./saved_models/p-{p[idx]}_n-{n}_K-{cmd_args.K}_s-{s[idx]}_s0-{s0[idx]}_d-{d[idx]}_' \
               f'w_range_l-0.5_w_range_u-2.0/GOD/'
        for size in sizes:
            A = np.zeros((cmd_args.K, p[idx], p[idx]))
            for gg in range(int(cmd_args.K / 8 / size)):
                dir = os.path.join(root, f'{size}_{8*gg*size}-{8*(gg+1)*size}.pkl')
                with open(dir, 'rb') as handle:
                    tt, A_est = pickle.load(handle)
                A[gg*size*8:(gg+1)*size*8] = A_est
                t[size][n].extend(tt)
            for k in range(cmd_args.K):
                G_true = np.abs(np.sign(db.G[k]))
                G_est = np.abs(A[k])
                G_est[G_est < cmd_args.threshold] = 0
                G_est = np.sign(G_est)
                print(G_est.sum())
                r = count_accuracy(G_true, G_est)
                print(r)
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
            temp_t = list(t[size][n]).copy()
            t[size][n] = np.mean(temp_t)
            fdr[size][n] = np.mean(temp_fdr)
            tpr[size][n] = np.mean(temp_tpr)
            fpr[size][n] = np.mean(temp_fpr)
            shd[size][n] = np.mean(temp_shd)
            nnz[size][n] = np.mean(temp_nnz)
            t_std[size][n] = np.std(temp_t)
            fdr_std[size][n] = np.std(temp_fdr)
            tpr_std[size][n] = np.std(temp_tpr)
            fpr_std[size][n] = np.std(temp_fpr)
            shd_std[size][n] = np.std(temp_shd)
            nnz_std[size][n] = np.std(temp_nnz)
    t = np.array([[t[size][n] for n in n_samples] for size in sizes])
    fdr = np.array([[fdr[size][n] for n in n_samples] for size in sizes])
    tpr = np.array([[tpr[size][n] for n in n_samples] for size in sizes])
    fpr = np.array([[fpr[size][n] for n in n_samples] for size in sizes])
    shd = np.array([[shd[size][n] for n in n_samples] for size in sizes])
    nnz = np.array([[nnz[size][n] for n in n_samples] for size in sizes])

    t_std = np.array([[t_std[size][n] for n in n_samples] for size in sizes])
    fdr_std = np.array([[fdr_std[size][n] for n in n_samples] for size in sizes])
    tpr_std = np.array([[tpr_std[size][n] for n in n_samples] for size in sizes])
    fpr_std = np.array([[fpr_std[size][n] for n in n_samples] for size in sizes])
    shd_std = np.array([[shd_std[size][n] for n in n_samples] for size in sizes])
    nnz_std = np.array([[nnz_std[size][n] for n in n_samples] for size in sizes])
    print(f'### time for p = {p[idx]} ###')
    for k in range(len(sizes)):
        log = f'k={int(2 ** k)} & '
        for l in range(len(n_samples)):
            log += f'${t[k][l]:.0f}\pm{t_std[k][l]:.0f}$ & '
        print(log[:-2] + '\\\\')
    print(f'### fdr for p = {p[idx]} ###')
    for k in range(len(sizes)):
        log = f'k={int(2**k)} & '
        for l in range(len(n_samples)):
            log += f'${fdr[k][l]:.3f}\pm{fdr_std[k][l]:.3f}$ & '
        print(log[:-2] + '\\\\')
    print(f'### tpr for p = {p[idx]} ###')
    for k in range(len(sizes)):
        log = f'k={int(2**k)} & '
        for l in range(len(n_samples)):
            log += f'${tpr[k][l]:.3f}\pm{tpr_std[k][l]:.3f}$ & '
        print(log[:-2] + '\\\\')
    print(f'### fpr for p = {p[idx]} ###')
    for k in range(len(sizes)):
        log = f'k={int(2**k)} & '
        for l in range(len(n_samples)):
            log += f'${fpr[k][l]:.3f}\pm{fpr_std[k][l]:.3f}$ & '
        print(log[:-2] + '\\\\')
    print(f'### shd for p = {p[idx]} ###')
    for k in range(len(sizes)):
        log = f'k={int(2**k)} & '
        for l in range(len(n_samples)):
            log += f'${shd[k][l]:.0f}\pm{shd_std[k][l]:.0f}$ & '
        print(log[:-2] + '\\\\')
    print(f'### nnz for p = {p[idx]} ###')
    for k in range(len(sizes)):
        log = f'k={int(2**k)} & '
        for l in range(len(n_samples)):
            log += f'${nnz[k][l]:.0f}\pm{nnz_std[k][l]:.0f}$ & '
        print(log[:-2] + '\\\\')
