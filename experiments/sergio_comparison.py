import os
import numpy as np
import matplotlib.pyplot as plt
import pickle
from multidag.common.cmd_args import cmd_args
from multidag.data_generator import Dataset
from multidag.dag_utils import count_accuracy, is_dag
from multidag.model import G_DAG
from multidag.sergio_dataset import SergioDataset

# Load recovered graphs
title = ['fdr', 'tpr', 'fpr', 'shd', 'nnz']
result_dir = 'sergio_results'
db = SergioDataset(cmd_args.real_dir)
home_dir = 'saved_models' if not cmd_args.baseline else 'saved_baselines'
model_type = home_dir.split('_')[1][:-1]
sim_file = cmd_args.real_dir.split('/')[-1]
multidag_result = []
with open(f'./{home_dir}/{sim_file}/rho-{cmd_args.rho}_lambda-{cmd_args.ld}_'
          f'c-{cmd_args.c}_gamma-{cmd_args.gamma}_'
          f'eta-{cmd_args.eta}_mu-{cmd_args.mu}_dual_interval-{cmd_args.dual_interval}/multidag_'
          f'group_size-{db.K}-{0}-{db.K}.pkl', 'rb') as handle:
    model = pickle.load(handle)[0]

# Measure recovery accuracy
K_mask = np.arange(0, db.K)
G_true = np.abs(np.sign(db.G[K_mask]))
G_est = np.abs(model['G'] * model['T'])
G_est[G_est < cmd_args.threshold] = 0
# G_est = np.sign(G_est)
for k in range(G_true.shape[0]):
    # threshold G_est before count_accuracy
    G_k = G_est[k]
    d = len(G_k)
    threshold = cmd_args.threshold
    if not is_dag(G_k):
        for val in np.sort(G_k[G_k > 0].flatten()):
            G_k[G_k <= val] = 0
            if is_dag(G_k):
                threshold = val
                break
    G_k = np.sign(G_k)
    r = count_accuracy(G_true[k], G_k)
    multidag_result.append([r[key] for key in r])

# Make results dir if needed
if not os.path.exists(f'{result_dir}/recovery_data.txt'):
    os.makedirs(f'{result_dir}/figures', exist_ok=True)
    line = f"model_type K n rho threshold ld {' '.join(title)}"
    result_file = open(f"{result_dir}/recovery_data.txt", 'w')
    result_file.write(line + '\n')

# Save a representative figure
fig_title = f"type-{model_type}_K-{db.K}_n-{db.n}_rho-{cmd_args.rho}_threshold-{cmd_args.threshold}_lambda-{cmd_args.ld}"
fig, axs = plt.subplots(1, 2)
fig.suptitle(fig_title)
axs[0].imshow(G_true[0].T)
axs[0].set_title('True')
axs[0].axis('off')
axs[1].imshow(np.sign(G_est[0]).T)
axs[1].set_title('Recovered')
axs[1].axis('off')
plt.savefig(f'{result_dir}/figures/{fig_title}.png', dpi=100)

# Save result metrics
mean_multidag_result = np.array(multidag_result).mean(axis=0)
print(f'###### multidag average results for K-{db.K} n-{db.n} #####')
for k, t in enumerate(title):
    print(f'{t}: {mean_multidag_result[k]:.6f}')
result_file = open(f'{result_dir}/recovery_data.txt', 'a')
line = f"{model_type} {db.K} {db.n} {cmd_args.rho} {cmd_args.threshold} {cmd_args.ld} {' '.join(mean_multidag_result.astype(str).tolist())}"
result_file.write(line + '\n')
