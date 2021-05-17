import torch.nn as nn
import torch
from vae4dag.common.consts import DEVICE, NONLINEARITIES
from vae4dag.common.utils import weights_init, MLP, hard_threshold, diff_hard_threshold, MLP_Batch
from torch.nn.parameter import Parameter
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn import LayerNorm
import torch.nn.functional as F
import math


class W_DAG(nn.Module):
    def __init__(self, num_dags, d):
        super(W_DAG, self).__init__()
        self.w = Parameter(torch.rand(size=[num_dags, d, d]))
        weights_init(self)

    def forward(self, idx):
        return self.w[idx]


class Encoder(nn.Module):
    """
    X to W
    """
    def __init__(self, d, tf_nhead: int = 8, tf_num_stacks: int = 6, tf_ff_dim: int = 64, tf_dropout: float = 0.0,
                 tf_act: str = 'relu', mlp_dim: str = '16-16-16', mlp_act='relu'):
        super(Encoder, self).__init__()

        dim_list = tuple(map(int, mlp_dim.split("-")))
        mlp_1st_dim = dim_list[0]
        self.mlp_out_dim = dim_list[-1]

        # Part 1: MLP is variable-wise, not shared across variables
        self.variable_mlp = MLP_Batch(d=d, input_dim=1, hidden_dims=mlp_dim, nonlinearity=mlp_act, act_last=None)

        weights_init(self)

        # Part 2: Sequence to Sequence Model (Transformer Encoder)

        encoder_layer = TransformerEncoderLayer(d_model=self.mlp_out_dim,
                                                nhead=tf_nhead,
                                                dim_feedforward=tf_ff_dim,
                                                dropout=tf_dropout,
                                                activation=tf_act)

        encoder_norm = LayerNorm(self.mlp_out_dim)

        self.tf_encoder = TransformerEncoder(encoder_layer, tf_num_stacks, encoder_norm)

        # Part 3: Pooling (no parameters)

        # Part 4: Sequence to Matrix
        # self.pairwise_score = PairwiseScore(dim_in=2 * self.mlp_out_dim, act='tanh')
        self.pairwise_score_pos = PairwiseScore(dim_in=self.mlp_out_dim, act='relu') # 'tanh')
        self.pairwise_score_neg = PairwiseScore(dim_in=self.mlp_out_dim, act='relu') #'tanh')
        # W_ij = u^T tanh(W1 Enc_i + W2 Enc_j)

        # Part 5: Threshold
        self.S_pos = Parameter(torch.ones(size=[d, d]) * 0.1)
        self.S_neg = Parameter(torch.ones(size=[d, d]) * 0.1)

    def forward(self, X):
        """
        :param X: [batch_size, n, d] tensor
        :return: W: [batch_size, d, d] tensor
        """
        assert len(X.shape) == 3
        batch_size, n, d = X.shape[0], X.shape[1], X.shape[2]

        # Part 1: MLP (shared layers)
        X_resize = X.view(batch_size * n, d, 1)  # size : [batch_size * n, d, 1]
        X_embed = self.variable_mlp(X_resize)  # size : [batch_size * n, d, mlp_out_dim]

        # Part 2: Seq to Seq (Transformer)
        X_in = X_embed.transpose(0, 1)  # [d, batch_size * n, mlp_out_dim]
        Seq_Enc = self.tf_encoder(X_in).transpose(0, 1).view(batch_size, n, d, self.mlp_out_dim)

        # Part 3: Pooling
        mean_pooling = torch.mean(Seq_Enc, dim=1)  # [batch_size, d, self.mlp_out_dim]

        # Part 4: Get adjacancy matrix
        # W_ij = u^T relu(W1 Enc_i + W2 Enc_j)

        W_pos = self.pairwise_score_pos(mean_pooling)
        W_neg = self.pairwise_score_neg(mean_pooling)

        # Part 5: Take threshold
        W_pos = F.relu(F.relu(W_pos) - self.S_pos ** 2)
        W_neg = F.relu(F.relu(W_neg) - self.S_neg ** 2)
        W = W_pos - W_neg

        # W_hard = hard_threshold(F.relu(self.S), W)
        # W_approx = diff_hard_threshold(F.relu(self.S), W, self.k)
        #
        # return (W_hard - W_approx).detach() + W_approx

        return W


class Decoder(nn.Module):
    
    def __init__(self, d, f_hidden_dims='16-16-1', f_act='relu', g_hidden_dims='16-16-1', g_act='relu', learn_sd=False):
        super(Decoder, self).__init__()

        dim_list = tuple(map(int, f_hidden_dims.split("-")))
        assert dim_list[-1] == 1
        dim_list = tuple(map(int, g_hidden_dims.split("-")))
        assert dim_list[-1] == 1

        self.f = MLP_Batch(d=d, input_dim=d, hidden_dims=f_hidden_dims, nonlinearity=f_act, act_last=None)
        if learn_sd:
            self.g = MLP_Batch(d=d, input_dim=d, hidden_dims=g_hidden_dims, nonlinearity=g_act, act_last=None)
        else:
            self.g = None

    def forward(self, W, X):
        """
        :param W: [batch, d, d] tensor  d1 == d
        :param X: [batch, n, d] tensor
        """
        batch, n, d = X.shape
        WX = torch.einsum('bji,bnj->bnij', W, X)  # [batch, n, d, d] tensor
        mean = self.f(WX.view(batch*n, d, d)).view(batch, n, d)  # [batch, n, d]
        if self.g is None:
            sd = None
        else:
            sd = torch.abs(self.g(WX.view(batch*n, d, d)).view(batch, n, d))
        return mean, sd

    def NLL(self, W, X):
        """
        negative log likelihood
        """
        mean, sd = self.forward(W, X)  # [batch, n, d]
        if sd is None:
            sd = 1.0
            log_z = 0.5 * math.log(2 * math.pi)
        else:
            log_z = 0.5 * math.log(2 * math.pi) + torch.log(sd)
        neg_log_likelihood = log_z + 0.5 * ((X - mean) / sd) ** 2  # [batch, n, d]

        return neg_log_likelihood


class PairwiseScore(nn.Module):
    def __init__(self, dim_in, act='tanh'):
        super(PairwiseScore, self).__init__()
        self.W1 = Parameter(torch.zeros(size=[dim_in, dim_in]))
        self.W2 = Parameter(torch.zeros(size=[dim_in, dim_in]))
        self.v = Parameter(torch.zeros(size=[dim_in]))
        self.activation = NONLINEARITIES[act]
        weights_init(self)

    def forward(self, X):
        """
        :param X: [batch_size, d, dim] tensor
        :return [batch_size, d, d] tensor

        W_ij = v^T tanh(W1 X_i + W2 X_j)
        """
        assert len(X.shape) == 3

        W1_Xi = F.linear(X, self.W1, bias=None)  # [batch_size, d, dim]
        W2_Xj = F.linear(X, self.W2, bias=None)

        W1_Xi_add_W2_Xj = W1_Xi.transpose(-1, -2).unsqueeze(-1) + W2_Xj.transpose(-1, -2).unsqueeze(-1).transpose(-1, -2)
        H_ij = self.activation(W1_Xi_add_W2_Xj)  # [batch_size, dim, d, d]

        W = torch.einsum('bijk,i->bjk', H_ij, self.v)  # [batch_size, d, d]

        return W


if __name__ == '__main__':
    EncNet = Encoder(d=5).to(DEVICE)
    X = torch.rand([2, 3, 5]).to(DEVICE)
    W = EncNet(X)  # [2, 5, 5]
    print(EncNet(X))

    # batch = 2
    # n = 3
    # d = 5
    # wx = torch.zeros(size=[batch, n, d, d])
    # for b in range(batch):
    #     for k in range(n):
    #         for i in range(d):
    #             wx[b, k, i] = W[b, :, i] * X[b, k]
    # print(wx)
    # print(torch.einsum('bji,bnj->bnij', W, X))

    DecNet = Decoder(d=5)
    nll = DecNet.NLL(W, X)
    print(nll)
