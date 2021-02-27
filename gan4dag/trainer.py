import torch
from tqdm import tqdm
from gan4dag.common.consts import DEVICE, OPTIMIZER
import math


D_Loss = torch.nn.BCEWithLogitsLoss(reduction='mean')


class LsemTrainer:
    def __init__(self, g_net, d_net, g_optimizer, d_optimizer, data_base, save_itr=500):
        self.g_net = g_net
        self.d_net = d_net
        self.db = data_base
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.train_itr = 0
        self.save_itr = save_itr

    def _train_epoch(self, epoch, tot_epoch, batch_size, progress_bar, dsc):
        self.g_net.train()
        self.d_net.train()
        data_loader = self.db.load_data(batch_size=batch_size,
                                        auto_reset=False,
                                        shuffle=True,
                                        device=DEVICE)
        num_iterations = len(range(0, self.db.num_dags, batch_size))

        total_d_loss = 0.0
        total_g_loss = 0.0
        g_loss_batch = 0.0
        num_invalid_W = 0
        for it, X in enumerate(data_loader):

            m = X.shape[0]  # number of DAGs in this batch
            n = X.shape[1]  # number of observed samples for each DAG

            # ---------------------
            #  Train Discriminator
            # ---------------------

            self.d_optimizer.zero_grad()

            # generate fake samples [m, n, d]
            X_fake, k = self.g_net.gen_batch_X(batch_size=m, n=n)
            num_invalid_W += k

            # compute loss
            score_fake = self.d_net(X_fake.detach())  # [m]
            score_real = self.d_net(X)  # [m]
            loss_fake = D_Loss(score_fake, torch.zeros(size=[m]).to(DEVICE))
            loss_real = D_Loss(score_real, torch.ones(size=[m]).to(DEVICE))
            d_loss = (loss_fake + loss_real) / 2

            # backward
            d_loss.backward()
            self.d_optimizer.step()

            d_loss_batch = d_loss.item()
            total_d_loss += d_loss_batch
            progress_bar.set_description("[Epoch %.2f] [D: %.3f] [G: %.3f] [invalid W: %d]" %
                                         (epoch + float(it + 1) / num_iterations, d_loss_batch, g_loss_batch,
                                          num_invalid_W) + dsc)

            # -----------------
            #  Train Generator
            # -----------------

            self.g_optimizer.zero_grad()

            # maximize log(sigmoid(D(G(z))))
            score_fake = self.d_net(X_fake)
            g_loss = D_Loss(score_fake, torch.ones(size=[m]).to(DEVICE))

            # backward
            g_loss.backward()
            self.g_optimizer.step()

            g_loss_batch = g_loss.item()
            total_g_loss += g_loss_batch
            progress_bar.set_description("[Epoch %.2f] [D: %.3f] [G: %.3f] [invalid W: %d]" %
                                         (epoch + float(it + 1) / num_iterations, d_loss_batch, g_loss_batch,
                                          num_invalid_W) + dsc)

            # -----------------
            #  Save
            # -----------------
            self.train_itr += 1
            if (self.train_itr % self.save_itr == 0) or ((epoch == tot_epoch-1) and (it == num_iterations-1)):
                self.save()
        return

    def save(self):
        # TODO: save network or some stats
        return

    def train(self, epochs, batch_size):
        """
        training logic
        """
        progress_bar = tqdm(range(epochs))
        dsc = ''
        print('*** Start training ***')
        for e in progress_bar:
            self._train_epoch(e, epochs, batch_size, progress_bar, dsc)
            # torch.save(self.algo_net.state_dict(), self.algo_model_dump)

        return


if __name__ == '__main__':
    from gan4dag.common.cmd_args import cmd_args
    from gan4dag.data_generator import LsemDataset
    import random
    import numpy as np

    random.seed(cmd_args.seed)
    np.random.seed(cmd_args.seed)
    torch.manual_seed(cmd_args.seed)

    num_dag = cmd_args.num_dag
    num_sample = cmd_args.num_sample
    threshold = cmd_args.threshold
    sparsity = cmd_args.sparsity
    d = cmd_args.d

    # ---------------------
    #  Synthetic Dataset
    # ---------------------

    # TODO: ground-truth meta-distribution
    W_mean = np.random.normal(size=[d, d]).astype(np.float32) * 3
    W_sd = np.random.rand(d, d).astype(np.float32)
    noise_mean = np.zeros(d, dtype=np.float32)
    noise_sd = np.ones(d, dtype=np.float32)

    print('*** Loading data ***')

    db = LsemDataset(W_mean, W_sd, sparsity, threshold, noise_mean, noise_sd, num_dag, num_sample)

    # ---------------------
    #  Initialize Networks
    # ---------------------
    from gan4dag.gan_model import GenNet, DiscNet

    print('*** Initializing networks ***')
    gen_net = GenNet(d=d).to(DEVICE)
    disc_net = DiscNet(d=d,
                       f_hidden_dims=cmd_args.f_hidden_dim,
                       f_nonlinearity=cmd_args.f_act,
                       output_hidden_dims=cmd_args.output_hidden_dim,
                       output_nonlinearity=cmd_args.output_act).to(DEVICE)

    # ---------------------
    #  Optimizer
    # ---------------------

    g_opt = OPTIMIZER[cmd_args.g_optimizer](gen_net.parameters(),
                                            lr=cmd_args.g_lr,
                                            weight_decay=cmd_args.weight_decay)
    d_opt = OPTIMIZER[cmd_args.d_optimizer](disc_net.parameters(),
                                            lr=cmd_args.d_lr,
                                            weight_decay=cmd_args.weight_decay)

    # ---------------------
    #  Trainer
    # ---------------------
    trainer = LsemTrainer(gen_net, disc_net, g_opt, d_opt, db, save_itr=cmd_args.save_itr)
    trainer.train(epochs=cmd_args.num_epochs, batch_size=cmd_args.batch_size)
