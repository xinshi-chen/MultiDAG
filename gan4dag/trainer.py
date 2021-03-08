import torch
from tqdm import tqdm
from gan4dag.common.consts import DEVICE
import os

D_Loss = torch.nn.BCEWithLogitsLoss(reduction='mean')


class LsemTrainer:
    def __init__(self, g_net, d_net, g_optimizer, d_optimizer, data_base, num_sample_gen, save_dir, model_dump, save_itr=500):
        self.g_net = g_net
        self.d_net = d_net
        self.db = data_base
        self.g_optimizer = g_optimizer
        self.d_optimizer = d_optimizer
        self.train_itr = 0
        self.save_itr = save_itr
        self.num_sample_gen = num_sample_gen

        self.model_dump = model_dump

        data_hp = 'LSEM-d-%d-ts-%.2f-sp-%.2f' % (self.db.d, self.db.W_threshold, self.db.W_sparsity)
        self.save_dir = save_dir + '/' + data_hp
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

    def _train_epoch(self, epoch, tot_epoch, batch_size, progress_bar, dsc, baseline=False):
        self.g_net.train()
        self.d_net.train()
        data_loader = self.db.load_data(batch_size=batch_size,
                                        auto_reset=False,
                                        shuffle=True,
                                        device=DEVICE,
                                        baseline=baseline)
        num_iterations = len(range(0, self.db.num_dags, batch_size))

        total_d_loss = 0.0
        total_g_loss = 0.0
        g_loss_batch = 0.0
        num_invalid_W = 0

        for it, X in enumerate(data_loader):

            m = X.shape[0]  # number of DAGs in this batch

            # ---------------------
            #  Train Discriminator
            # ---------------------

            self.d_optimizer.zero_grad()

            # generate fake samples [m, n, d]
            if baseline:
                X_fake, _, k = self.g_net.gen_batch_dag(batch_size=m)
            else:
                X_fake, k = self.g_net.gen_batch_X(batch_size=m, n=self.num_sample_gen)
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
            last_itr = (self.train_itr == tot_epoch * num_iterations)
            if (self.train_itr % self.save_itr == 0) or last_itr:
                self.save(self.train_itr, baseline)
        return

    def save(self, itr,  baseline=False):
        if baseline:
            save_dir = self.save_dir + '/baseline'
            if not os.path.isdir(save_dir):
                os.makedirs(save_dir)
            dump = save_dir + '/Itr-%d-' % itr + self.model_dump
        else:
            dump = self.save_dir + '/Itr-%d-' % itr + self.model_dump
        torch.save(self.g_net.state_dict(), dump)
        dump = dump[:-5] + '_disc.dump'
        torch.save(self.d_net.state_dict(), dump)

    def load(self, itr, baseline=False):
        if baseline:
            save_dir = self.save_dir + '/baseline'
            dump = save_dir + '/Itr-%d-' % itr + self.model_dump
        else:
            dump = self.save_dir + '/Itr-%d-' % itr + self.model_dump

        self.g_net.load_state_dict(torch.load(dump))
        dump = dump[:-5] + '_disc.dump'
        self.d_net.load_state_dict(torch.load(dump))

    def train(self, epochs, batch_size, baseline=False, start_epoch=0):
        """
        training logic
        """

        if start_epoch > 0:
            completed_epochs = start_epoch
            num_itr_per_epoch = len(range(0, self.db.num_dags, batch_size))
            itr = num_itr_per_epoch * completed_epochs
            self.load(itr, baseline)

        progress_bar = tqdm(range(start_epoch, start_epoch + epochs))
        dsc = ''
        print('*** Start training ***')
        for e in progress_bar:
            self._train_epoch(e, epochs, batch_size, progress_bar, dsc, baseline)

        return
