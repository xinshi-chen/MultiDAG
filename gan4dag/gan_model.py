import torch.nn as nn
import torch
from gan4dag.common.consts import DEVICE
from gan4dag.common.utils import weights_init, MLP
from gan4dag.dag_utils import project_to_dag, sampler
from torch.nn.parameter import Parameter


class GenNet(nn.Module):

    def __init__(self, d, noise_mean=None, noise_sd=None, W_sd=None):
        super(GenNet, self).__init__()
        self.d = d
        self.W = Parameter(torch.empty(size=[d, d]))

        if noise_mean is None:
            self.noise_mean = Parameter(torch.zeros(size=[d]))
        else:
            self.noise_mean = torch.tensor(noise_mean).to(DEVICE)

        weights_init(self)

        if W_sd is None:
            self.V = Parameter(torch.ones(size=[d, d]))
        else:
            self.V = torch.tensor(W_sd).to(DEVICE)

        if noise_sd is None:
            self.noise_sd = Parameter(torch.ones(size=[d]))
        else:
            self.noise_sd = torch.tensor(noise_sd).to(DEVICE)

    def forward(self, Z):
        """
        :param Z: [d, d] or [m, d, d] tensor
        :return:
        """
        W = Z * self.V
        W = W + self.W

        return W

    def gen_one_dag(self, diff=True):
        num_passed_z = 0

        # (1) Generate z -> W; and compute projection matrix P.
        # Do not differentiate through this part.
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

        if not diff:
            return w, P, num_passed_z

        # (2) z -> W; W -> W * P
        # This part is differentiable
        W = self.forward(z.detach())
        W = W * P  # make it a DAG
        # TODO: should have one more layer to adjust the weight, in case the sparsity/threshold is not correct
        return W, P, num_passed_z

    def gen_batch_dag(self, batch_size, diff=True):
        num_passed_z = 0

        W = torch.zeros(size=[batch_size, self.d, self.d]).to(DEVICE)
        P = torch.zeros(size=[batch_size, self.d, self.d]).to(DEVICE)
        for i in range(batch_size):
            W[i], P[i], k = self.gen_one_dag(diff)
            num_passed_z += k
        return W, P, num_passed_z

    def gen_one_X(self, n, diff=True):

        # generate a DAG
        W, P, num_passed_z = self.gen_one_dag(diff)

        # samples
        X = sampler(W, n, self.noise_mean, self.noise_sd, noise_type='gauss')
        return X, num_passed_z

    def gen_batch_X(self, batch_size, n, diff=True):
        num_passed_z = 0
        X = torch.zeros(size=[batch_size, n, self.d]).to(DEVICE)
        # TODO: speed up?
        for i in range(batch_size):
            X[i], k = self.gen_one_X(n, diff)
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


# for baseline
class DiscGIN(nn.Module):
    """
    Follow the architecture of GIN
    """
    def __init__(self, d,
                 node_f_dim=32,
                 hidden_dims='32-32',
                 nonlinearity='relu',
                 output_hidden_dims='64-1',
                 output_nonlinearity='relu',
                 hops=3):
        """

        :param d:
        :param node_f_dim: node feature dimension
        :param f_hidden_dims:
        :param f_nonlinearity:
        :param output_hidden_dims:
        :param output_nonlinearity:
        """
        super(DiscGIN, self).__init__()

        self.d = d
        self.node_f_dim = node_f_dim
        self.hops = hops

        self.init_features = Parameter(torch.empty(size=[d, node_f_dim]))

        # make sure the output dim of MLP is node_f_dim
        hidden_dims_list = tuple(map(int, hidden_dims.split("-")))
        if hidden_dims_list[-1] != node_f_dim:
            hidden_dims += '-' + str(node_f_dim)

        # MLP for transform messages
        mlp_list = []
        for i in range(hops):
            mlp_list.append(MLP(input_dim=node_f_dim,
                                hidden_dims=hidden_dims,
                                nonlinearity=nonlinearity,
                                act_last=nonlinearity))
        self.mlp = nn.ModuleList(mlp_list)

        # Output layer
        input_dim = hops * node_f_dim
        # make sure the output dim of MLP is 1
        hidden_dims_list = tuple(map(int, output_hidden_dims.split("-")))
        if hidden_dims_list[-1] != 1:
            hidden_dims += '-1'

        self.output_layer = MLP(input_dim=input_dim,
                                hidden_dims=output_hidden_dims,
                                nonlinearity=output_nonlinearity,
                                act_last=None).to(DEVICE)

        weights_init(self)
        self.epsilon = Parameter(torch.ones(size=[hops]) * 0.5)

    def forward(self, W):
        """

        :param W: [m, d, d] tensor. m is batch size. d is variable dimension.
        :return: scores of dimension [m]
        """
        m = W.shape[0]
        node_features = torch.zeros(size=[m, self.d, self.node_f_dim]).to(DEVICE)  # [m, d, node_f_dim]
        readout = torch.zeros(size=[self.hops, m, self.node_f_dim]).to(DEVICE)  # [m, hops, node_f_dim]
        for i in range(self.hops):

            # -----------------
            #  Aggregate features
            # -----------------
            if i == 0:
                pa_message = torch.matmul(W.transpose(-1, -2), self.init_features)  # [m, d, node_f_dim]
                combined_features = pa_message + self.init_features * (1 + self.epsilon[i])  # [m, d, node_f_dim]
            else:
                pa_message = torch.einsum('bij,bjk->bik', W.transpose(-1, -2), node_features)  # batch matmul
                combined_features = pa_message + node_features * (1 + self.epsilon[i])

            # -----------------
            #  Transform
            # -----------------
            node_features = self.mlp[i](combined_features)  # [m, d, node_f_dim]
            readout[i] = torch.sum(node_features, dim=1)  # [m, node_f_dim]

        readout = readout.transpose(1, 0).reshape(m, self.hops * self.node_f_dim)  # [m, hops * node_f_dim]
        score = self.output_layer(readout)  # [m, 1]

        return score.view(-1)


if __name__ == '__main__':
    d = 5
    gen = GenNet(d).to(DEVICE)
    X_gen, _ = gen.gen_batch_X(batch_size=10, n=5)

    disc = DiscNet(d).to(DEVICE)
    scores = disc(X_gen)
    print(scores)
