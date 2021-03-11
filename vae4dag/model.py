import torch.nn as nn
import torch
from vae4dag.common.consts import DEVICE, NONLINEARITIES
from vae4dag.common.utils import weights_init, MLP, hard_threshold, diff_hard_threshold
from torch.nn.parameter import Parameter
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.nn import LayerNorm
import torch.nn.functional as F


class Encoder(nn.Module):
    """
    X to W
    """
    def __init__(self, d, tf_nhead: int = 8, tf_num_stacks: int = 6, tf_ff_dim: int = 64, tf_dropout: float = 0.0,
                 tf_act: str = 'relu', mlp_dim: str = '16-16-16', mlp_act='relu', temperature=3.0):
        super(Encoder, self).__init__()

        self.k = temperature

        dim_list = tuple(map(int, mlp_dim.split("-")))
        mlp_1st_dim = dim_list[0]
        self.mlp_out_dim = dim_list[-1]

        # Part 0: First layer of MLP is variable-wise, not shared
        self.W_1st = Parameter(torch.zeros(size=[d, mlp_1st_dim]))
        self.bias_1st = Parameter(torch.zeros(size=[d, mlp_1st_dim]))
        self.act_1st = NONLINEARITIES[mlp_act]

        # Part 1: Shared MLP

        mlp_dim = "-".join(map(str, dim_list[1:]))
        self.mlp = MLP(input_dim=mlp_1st_dim,
                       hidden_dims=mlp_dim,
                       nonlinearity=mlp_act,
                       act_last=None).to(DEVICE)

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
        self.pairwise_score = PairwiseScore(dim_in=2 * self.mlp_out_dim, act='tanh')
        # W_ij = u^T tanh(W1 Enc_i + W2 Enc_j)

        # Part 5: Threshold
        self.S = Parameter(torch.ones(size=[d, d]) * 0.1)

    def forward(self, X):
        """
        :param X: [batch_size, n, d] tensor
        :return: W: [batch_size, d, d] tensor
        """
        assert len(X.shape) == 3
        batch_size, n, d = X.shape[0], X.shape[1], X.shape[2]

        # Part 0: Variable-wise 1st layer
        XW_b = X.unsqueeze(-1) * self.W_1st + self.bias_1st  # [batch_size, n, d, hidden_dim]
        H = self.act_1st(XW_b)

        # Part 1: MLP (shared layers)
        X_embed = self.mlp(H)  # size : [batch_size, n, d, out_dim]

        # Part 2: Seq to Seq (Transformer)
        X_in = X_embed.view(batch_size * n, d, self.mlp_out_dim).transpose(0, 1)  # [d, batch_size * n, self.mlp_out_dim]
        Seq_Enc = self.tf_encoder(X_in).transpose(0, 1).view(batch_size, n, d, self.mlp_out_dim)

        # Part 3: Pooling
        mean_pooling = torch.mean(Seq_Enc, dim=1)  # [batch_size, d, self.mlp_out_dim]
        max_pooling, _ = torch.max(Seq_Enc, dim=1)
        X_pooling = torch.cat([mean_pooling, max_pooling], dim=-1)  # [batch_size, d, 2 * self.mlp_out_dim]

        # Part 4: Get adjacancy matrix
        W = self.pairwise_score(X_pooling)

        # Part 5: Take threshold
        W_hard = hard_threshold(self.S, W)
        W_approx = diff_hard_threshold(self.S, W, self.k)

        return (W_hard - W_approx).detach() + W_approx


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
    print(EncNet(X))
