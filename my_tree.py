import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class MyTree(nn.Module):
    def __init__(
        self, depth, in_dim, sub_features_rate, out_class, jointly_training=True
    ):
        super().__init__()
        self._depth = depth
        self._in_dim = in_dim
        self._sub_features_rate = sub_features_rate
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
        self._jointly_training = jointly_training
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
        while end_idx <= self.n_nodes:
            if start_idx == 0:
                root_layer_probs = nodes_prob[:, start_idx:end_idx]
                probs = torch.cat(
                    [root_layer_probs, 1 - root_layer_probs], dim=1
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

    def extra_repr(self) -> str:
        return "\n".join(
            [
                f"depth: {self.depth}",
                f"in_dim: {self.in_dim}",
                f"sub_features_rate: {self.sub_features_rate}",
                f"out_class: {self.n_class}",
                f"jointly_training: {self.jointly_training}",
            ]
        )

    @property
    def jointly_training(self):
        return self._jointly_training

    @property
    def depth(self):
        return self._depth

    @property
    def in_dim(self):
        return self._in_dim

    @property
    def sub_features_rate(self):
        return self._sub_features_rate

    @property
    def n_leaf(self):
        return self.pi.shape[0]

    @property
    def n_class(self):
        return self.pi.shape[1]

    @property
    def n_nodes(self):
        return self.n_leaf - 1
