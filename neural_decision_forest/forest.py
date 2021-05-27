import torch
import torch.nn as nn

from .tree import Tree

__all__ = ["Forest", "NeuralDecisionForest"]


class Forest(nn.Module):
    def __init__(
        self,
        n_tree,
        tree_depth,
        n_in_feature,
        tree_feature_rate,
        n_class,
        jointly_training,
        tree_cls=Tree,
    ):
        super(Forest, self).__init__()
        self.trees = nn.ModuleList()
        self.n_tree = n_tree
        for _ in range(n_tree):
            tree = tree_cls(
                tree_depth, n_in_feature, tree_feature_rate, n_class, jointly_training
            )
            self.trees.append(tree)

    def forward(self, x):
        probs = []
        for tree in self.trees:
            mu = tree(x)
            p = tree.cal_prob(mu, tree.get_pi())
            probs.append(p.unsqueeze(2))
        probs = torch.cat(probs, dim=2)
        prob = torch.sum(probs, dim=2) / self.n_tree

        return prob


class NeuralDecisionForest(nn.Module):
    def __init__(self, feature_layer, forest):
        super().__init__()
        self.feature_layer = feature_layer
        self.forest = forest

    def forward(self, x):
        out = self.feature_layer(x)
        out = out.view(x.size()[0], -1)
        out = self.forest(out)
        return out
