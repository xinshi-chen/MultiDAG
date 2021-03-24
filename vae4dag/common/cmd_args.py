import argparse
import os

cmd_opt = argparse.ArgumentParser(description='')

cmd_opt.add_argument('-gpu', type=int, default=-1, help='-1: cpu; 0 - ?: specific gpu index')
cmd_opt.add_argument('-seed', type=int, default=999983, help='seed')

# architecture

"""
encoder
"""
# 1. mlp
cmd_opt.add_argument('-mlp_hidden_dim', type=str, default='32-32-32')
cmd_opt.add_argument('-mlp_act', type=str, default='relu')
# 2. transformer
cmd_opt.add_argument('-tf_nhead', type=int, default=8)
cmd_opt.add_argument('-tf_num_stacks', type=int, default=1)
cmd_opt.add_argument('-tf_ff_dim', type=int, default=64)
cmd_opt.add_argument('-tf_act', type=str, default='relu')
# 3. threshold
cmd_opt.add_argument('-temperature', type=float, default=5.0)

"""
decoder
"""
cmd_opt.add_argument('-f_hidden_dim', type=str, default='1-16-1')
cmd_opt.add_argument('-f_act', type=str, default='relu')
cmd_opt.add_argument('-g_hidden_dim', type=str, default='16-16-1')
cmd_opt.add_argument('-g_act', type=str, default='relu')

# hyperparameters for synthetic distribution
cmd_opt.add_argument('-d', type=int, default=10, help='dimension of RV')
cmd_opt.add_argument('-num_sample', type=int, default=20, help='number of observed samples')
cmd_opt.add_argument('-p', type=float, default=0.5, help='proportion of samples used for structure recovery')
cmd_opt.add_argument('-num_sample_test', type=int, default=50, help='number of samples for test evaluation')

cmd_opt.add_argument('-threshold', type=float, default=0.1)
cmd_opt.add_argument('-sparsity', type=float, default=1.0, help='l1 penalty coefficient in the projection')
cmd_opt.add_argument('-true_f_hidden_dim', type=str, default='1-16-1')
cmd_opt.add_argument('-true_f_act', type=str, default='relu')

cmd_opt.add_argument('-learn_sd', type=eval, default=False, help='If false, then only learn the mean of the meta-distribution, while the variance is given.')

# hyperparameters for training
cmd_opt.add_argument('-num_train', type=int, default=512, help='number of DAGs')
cmd_opt.add_argument('-num_vali', type=int, default=64, help='number of DAGs')
cmd_opt.add_argument('-num_test', type=int, default=64, help='number of DAGs')

cmd_opt.add_argument('-batch_size', type=int, default=128, help='batch size')
cmd_opt.add_argument('-e_lr', type=float, default=1e-3, help='learning rate of encoder')
cmd_opt.add_argument('-d_lr', type=float, default=1e-3, help='learning rate of decoder')
cmd_opt.add_argument('-w_lr', type=float, default=1e-3, help='learning rate of W_DAG')

cmd_opt.add_argument('-weight_decay', type=float, default=1e-5)
cmd_opt.add_argument('-num_epochs', type=int, default=1000, help='num epochs')
cmd_opt.add_argument('-e_optimizer', type=str, default='adam')
cmd_opt.add_argument('-d_optimizer', type=str, default='adam')
cmd_opt.add_argument('-w_optimizer', type=str, default='adam')

cmd_opt.add_argument('-save_itr', type=int, default=8, help='how many iterations to save the trained states')
cmd_opt.add_argument('-start_epoch', type=int, default=0)

cmd_opt.add_argument('--hw_type', type=str, default='notears', choices=['notears', 'daggnn'])

cmd_opt.add_argument('-rho', type=float, default=0.1)
cmd_opt.add_argument('-alpha', type=float, default=1.0)
cmd_opt.add_argument('-ld', type=float, default=0.1)
cmd_opt.add_argument('-c', type=float, default=1.0)
cmd_opt.add_argument('-eta', type=float, default=0.01)

cmd_opt.add_argument('-phase', type=str, default='train')


# save
cmd_opt.add_argument('-save_dir', type=str, default='./saved_models', help='save folder')

cmd_args = cmd_opt.parse_args()
if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

print(cmd_args)
