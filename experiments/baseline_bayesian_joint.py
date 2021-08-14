import numpy as np
import os, sys
import torch
import pickle

from multidag.common.cmd_args import cmd_args
p=cmd_args.p
n=cmd_args.n_sample
K=cmd_args.K
s0=cmd_args.s0
s=cmd_args.s
d=cmd_args.d
w_range=(0.5, 2.0)
hp_dict = {'p': p,
           'n': n,
           'K': K,
           's': s,
           's0': s0,
           'd': d,
           'w_range_l': w_range[0],
           'w_range_u': w_range[1]}

hp = ''
for key in hp_dict:
    hp += key + '-' + str(hp_dict[key]) + '_'
hp = hp[:-1]
data_dir = os.path.join('../data', hp)
G = pickle.load(open(data_dir + '/DAGs.pkl', 'rb'))[0]
X = pickle.load(open(data_dir + '/samples_gid.pkl', 'rb')).numpy()
np.save(data_dir + '/G.npy', G)
np.save(data_dir + '/X.npy', X)
