import torch.nn as nn
import torch
from gan4dag.common.consts import DEVICE
from gan4dag.common.utils import weights_init, MLP
from gan4dag.dag_utils import project_to_dag, sampler
from torch.nn.parameter import Parameter


class GenNet(nn.Module):

    def __init__(self, d):
        super(GenNet, self).__init__()
        self.d = d
        self.W = Parameter(torch.empty(size=[d, d]))
        self.noise_mean = Parameter(torch.empty(size=[d]))
        weights_init(self)
        self.V = Parameter(torch.ones(size=[d, d]))
        self.noise_sd = Parameter(torch.ones(size=[d]))

    def forward(self, Z):
        """
        :param Z: [d, d] or [m, d, d] tensor
        :return:
        """
        W = Z * self.V
        W = W + self.W

        return W

    def gen_one_X(self, n):

        # (1) Generate z -> W; and compute projection matrix P.
        # Do not differentiate through this part.
        num_passed_z = 0
        with torch.no_grad():
            while 1:
                z = torch.normal(0, 1, size=(self.d, self.d)).float().to(DEVICE)
                w = self.forward(z).detach()
                _, P = project_to_dag(w.cpu().numpy(), sparsity=1.0, w_threshold=0.1)
                if P is not None:
                    P = torch.tensor(P).to(DEVICE).detach()
                    break
                else:
                    num_passed_z += 1

        # (2) z -> W; W -> W * P
        # This part is differentiable
        W = self.forward(z.detach())
        W = W * P  # make it a DAG
        # TODO: should have one more layer to adjust the weight, in case the sparsity/threshold is not correct

        # samples
        X = sampler(W, n, self.noise_mean, self.noise_sd, noise_type='gauss')
        return X, num_passed_z

    def gen_batch_X(self, batch_size, n):
        num_passed_z = 0
        X = torch.zeros(size=[batch_size, n, self.d]).to(DEVICE)
        # TODO: speed up?
        for i in range(batch_size):
            X[i], k = self.gen_one_X(n)
            num_passed_z += k
        return X, num_passed_z


class DiscNet(nn.Module):

    def __init__(self, d,
                 f_hidden_dims='32-32',
                 f_nonlinearity='relu',
                 output_hidden_dims='32-32-1',
                 output_nonlinearity='relu'):
        super(DiscNet, self).__init__()
        self.d = d

        # DeepSet based on f
        self.f = MLP(input_dim=d,
                     hidden_dims=f_hidden_dims,
                     nonlinearity=f_nonlinearity,
                     act_last=f_nonlinearity).to(DEVICE)

        # Output layer
        f_hidden_dims = tuple(map(int, f_hidden_dims.split("-")))
        input_dim = f_hidden_dims[-1]
        hidden_dims = tuple(map(int, output_hidden_dims.split("-")))
        if hidden_dims[-1] != 1:
            # make sure the output dim of MLP is 1
            output_hidden_dims += '-1'
        self.output_layer = MLP(input_dim=input_dim,
                                hidden_dims=output_hidden_dims,
                                nonlinearity=output_nonlinearity,
                                act_last=None).to(DEVICE)

        weights_init(self)

    def forward(self, X):
        """

        :param X: [m, n, d] tensor. m is batch size. n is the number of samples. d is variable dimension.
        :return: scores of dimension [m]
        """
        f_X = self.f(X)
        DS_X = torch.mean(f_X, dim=1)  # DeepSet: 1/n sum_{i=1}^n f(X_i)
        score = self.output_layer(DS_X)  # [m, 1]
        return score.view(-1)


if __name__ == '__main__':
    d = 5
    gen = GenNet(d).to(DEVICE)
    X_gen, _ = gen.gen_batch_X(batch_size=10, n=5)

    disc = DiscNet(d).to(DEVICE)
    scores = disc(X_gen)
    print(scores)
