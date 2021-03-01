import argparse
import os

cmd_opt = argparse.ArgumentParser(description='')

cmd_opt.add_argument('-d_model_dump', type=str, default='./scratch/d_model.dump')
cmd_opt.add_argument('-g_model_dump', type=str, default='./scratch/g_model.dump')
cmd_opt.add_argument('-gpu', type=int, default=-1, help='-1: cpu; 0 - ?: specific gpu index')
cmd_opt.add_argument('-seed', type=int, default=999983, help='seed')

# architecture
cmd_opt.add_argument('-f_hidden_dim', type=str, default='32-32')
cmd_opt.add_argument('-f_act', type=str, default='relu')
cmd_opt.add_argument('-output_hidden_dim', type=str, default='32-32')
cmd_opt.add_argument('-output_act', type=str, default='relu')

# hyperparameters for synthetic distribution
cmd_opt.add_argument('-d', type=int, default=5, help='dimension of RV')
cmd_opt.add_argument('-threshold', type=float, default=0.1)
cmd_opt.add_argument('-sparsity', type=float, default=1.0, help='l1 penalty coefficient in the projection')
cmd_opt.add_argument('-learn_noise', type=eval, default=False, help='If true, then noise distribution needs to be learned. Otherwise, it is assumed to be given.')

# hyperparameters for training
cmd_opt.add_argument('-num_dag', type=int, default=50, help='number of DAGs')
cmd_opt.add_argument('-num_sample', type=int, default=10, help='number of observed samples')
cmd_opt.add_argument('-batch_size', type=int, default=2, help='batch size')
cmd_opt.add_argument('-g_lr', type=float, default=1e-4, help='learning rate of generator')
cmd_opt.add_argument('-d_lr', type=float, default=1e-4, help='learning rate of discriminator')
cmd_opt.add_argument('-weight_decay', type=float, default=1e-5)
cmd_opt.add_argument('-num_epochs', type=int, default=100, help='num epochs')
cmd_opt.add_argument('-g_optimizer', type=str, default='adam')
cmd_opt.add_argument('-d_optimizer', type=str, default='adam')
cmd_opt.add_argument('-save_itr', type=int, default=500, help='how many iterations to save the trained states')

cmd_opt.add_argument('-num_sample_gen', type=int, default=100, help='number of observed samples for generator')

# not used yet
cmd_opt.add_argument('-save_dir', type=str, default='./scratch', help='save folder')
cmd_opt.add_argument('-phase', type=str, default='train', help='training or testing phase')
cmd_opt.add_argument('-save_file', type=str, default='./scratch/saveloss.pkl', help='filename for saving the losses')
cmd_opt.add_argument('-filename', type=str, default='results_2stage.pkl', help='filename for saving the results')
cmd_opt.add_argument('-save_loc', type=str, default=None, help='filename for saving the losses')

cmd_args = cmd_opt.parse_args()
if cmd_args.save_dir is not None:
    if not os.path.isdir(cmd_args.save_dir):
        os.makedirs(cmd_args.save_dir)

print(cmd_args)
