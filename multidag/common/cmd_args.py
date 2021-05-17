import argparse
import os

cmd_opt = argparse.ArgumentParser(description='')

cmd_opt.add_argument('-gpu', type=int, default=-1, help='-1: cpu; 0 - ?: specific gpu index')
cmd_opt.add_argument('-seed', type=int, default=999983, help='seed')


# hyperparameters for synthetic distribution
cmd_opt.add_argument('--real_dir', type=str, default='', help='dir of real data')
cmd_opt.add_argument('-p', type=int, default=20, help='dimension of RV')
cmd_opt.add_argument('-n_sample', type=int, default=30, help='number of observed samples')
cmd_opt.add_argument('-s', type=int, default=15, help='union support size')
cmd_opt.add_argument('-s0', type=int, default=10, help='support size of each DAG')
cmd_opt.add_argument('-K', type=int, default=16, help='number of tasks')
cmd_opt.add_argument('-d', type=int, default=10, help='number of parents')
cmd_opt.add_argument('--group_size', type=int, default=2, help='group size used in learning')
cmd_opt.add_argument('--group_start', type=int, default=0)

# hyperparameters for training
cmd_opt.add_argument('-batch_size', type=int, default=128, help='batch size')
cmd_opt.add_argument('-g_lr', type=float, default=1e-3, help='learning rate of G_DAG')

cmd_opt.add_argument('-weight_decay', type=float, default=1e-5)
cmd_opt.add_argument('-num_epochs', type=int, default=10000, help='num epochs')
cmd_opt.add_argument('-optimizer', type=str, default='adam')

cmd_opt.add_argument('-start_epoch', type=int, default=0)

cmd_opt.add_argument('--hw_type', type=str, default='notears', choices=['notears', 'daggnn'])

cmd_opt.add_argument('-rho', type=float, default=0.01)
cmd_opt.add_argument('-ld', type=float, default=1)
cmd_opt.add_argument('-c', type=float, default=1)
cmd_opt.add_argument('-eta', type=float, default=0.412)
cmd_opt.add_argument('-mu', type=float, default=1)
cmd_opt.add_argument('-gamma', type=float, default=1e-5)
# cmd_opt.add_argument('-gamma', type=float, default=0)
cmd_opt.add_argument('--dual_interval', type=int, default=50)
cmd_opt.add_argument('--threshold', type=float, default=0.3)

cmd_opt.add_argument('-phase', type=str, default='train')


# save
cmd_opt.add_argument('-save_dir', type=str, default='./saved_models', help='save folder')

cmd_args = cmd_opt.parse_args()
if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

print(cmd_args)
