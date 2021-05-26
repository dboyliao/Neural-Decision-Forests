from functools import reduce

import numpy as np
import torch
from torch import nn


def batch_kron_prod(a, b):
    assert a.ndim == 2
    assert b.ndim == 2
    kron = torch.einsum("bj,bk->bjk", a, b)
    out_dim = reduce(lambda a, b: a * b, kron.shape[1:], 1)
    return kron.view(-1, out_dim)


class MyTree(nn.Module):
    def __init__(
        self, depth, in_dim, out_class, sub_features_rate=0.5, jointly_training=True
    ):
        super().__init__()

        n_leaf = 2 ** depth
        n_nodes = 2 ** depth - 1
        n_subfeats = int(sub_features_rate * in_dim)
        # used features in this tree
        onehot = np.eye(in_dim)
        using_idx = np.random.choice(np.arange(in_dim), n_subfeats, replace=False)
        feature_mask = onehot[using_idx].T
        self.feature_mask = nn.Parameter(
            torch.from_numpy(feature_mask).type(torch.FloatTensor),
            requires_grad=False,
        )
        # leaf label distribution
        if jointly_training:
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
        while end_idx != self.n_nodes:
            current_layer_probs = nodes_prob[:, start_idx:end_idx]
            if start_idx == 0:
                probs = torch.cat([current_layer_probs, 1 - current_layer_probs], dim=1)
            else:
                prob_current_level = torch.cat(
                    [
                        nodes_prob[:, start_idx:end_idx, None],
                        1 - nodes_prob[:, start_idx:end_idx, None],
                    ],
                    dim=-1,
                ).view(batch_size, -1)
                probs = batch_kron_prod(probs, prob_current_level)
            start_idx = end_idx
            end_idx = end_idx * 2 + 1
        return probs

    @property
    def n_leaf(self):
        return self.pi.shape[0]

    @property
    def n_class(self):
        return self.pi.shape[1]

    @property
    def n_nodes(self):
        return self.n_leaf - 1
