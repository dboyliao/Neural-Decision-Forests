from functools import reduce

import numpy as np
import torch
from torch import einsum, nn
from torch.nn import functional as F


def batch_kron_prod(a, b):
    assert a.ndim == 2
    assert b.ndim == 2
    kron = einsum("bj,bk->bjk", a, b)
    out_dim = reduce(lambda a, b: a * b, kron.shape[1:], 1)
    return kron.view(-1, out_dim)


class MyTree(nn.Module):
    def __init__(
        self, depth, in_dim, sub_features_rate, out_class, jointly_training=True
    ):
        super().__init__()
        n_leaf = 2 ** depth
        n_nodes = 2 ** depth - 1
        n_subfeats = int(sub_features_rate * in_dim)
        # used features in this tree
        onehot = np.eye(in_dim)
        using_idx = np.random.choice(in_dim, n_subfeats, replace=False)
        feature_mask = onehot[using_idx].T
        self.feature_mask = nn.Parameter(
            torch.from_numpy(feature_mask).type(torch.FloatTensor),
            requires_grad=False,
        )
        # leaf label distribution
        self.jointly_training = jointly_training
        if self.jointly_training:
            pi = np.random.rand(n_leaf, out_class)
            self.pi = nn.Parameter(
                torch.from_numpy(pi).type(torch.FloatTensor), requires_grad=True
            )
        else:
            pi = np.ones((n_leaf, out_class)) / out_class
            self.pi = nn.Parameter(
                torch.from_numpy(pi).type(torch.FloatTensor), requires_grad=False
            )
        self.decision = nn.Sequential(nn.Linear(n_subfeats, n_nodes), nn.Sigmoid())

    def forward(self, x):
        batch_size = x.shape[0]
        features = torch.mm(x, self.feature_mask)
        nodes_prob = self.decision(features)  # B x num_nodes
        probs = 1
        start_idx = 0
        end_idx = 1
        while end_idx <= self.n_nodes:
            current_layer_probs = nodes_prob[:, start_idx:end_idx]
            if start_idx == 0:
                probs = torch.cat(
                    [current_layer_probs, 1 - current_layer_probs], dim=1
                )  # B x 2
            else:
                probs_this_level = torch.cat(
                    [
                        nodes_prob[:, start_idx:end_idx, None],
                        1 - nodes_prob[:, start_idx:end_idx, None],
                    ],
                    dim=-1,
                ).view(
                    batch_size, -1
                )  # B x (2 * probs.shape[1])
                probs = probs.repeat_interleave(2, dim=-1) * probs_this_level
            start_idx = end_idx
            end_idx = end_idx * 2 + 1
        return probs

    def get_pi(self):
        if self.jointly_training:
            return F.softmax(self.pi, dim=-1)
        else:
            return self.pi

    @staticmethod
    def cal_prob(mu, pi):
        """
        :param mu [batch_size,n_leaf]
        :param pi [n_leaf,n_class]
        :return: label probability [batch_size,n_class]
        """
        p = torch.mm(mu, pi)
        return p

    def update_pi(self, new_pi):
        self.pi.data = new_pi

    @property
    def n_leaf(self):
        return self.pi.shape[0]

    @property
    def n_class(self):
        return self.pi.shape[1]

    @property
    def n_nodes(self):
        return self.n_leaf - 1
