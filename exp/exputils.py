import numpy as np
import torch
import torch.nn.functional as F


class BinMaxMarginLoss:
    def __init__(self, pos_margin=1.0, neg_margin=1.0, pos_scale=1.0, neg_scale=1.0):
        self.pos_margin = pos_margin
        self.neg_margin = neg_margin
        self.pos_scale = pos_scale
        self.neg_scale = neg_scale

    def loss(self, labels_true, logits):
        tmp1 = torch.sum(labels_true * F.relu(self.pos_margin - logits)) * self.pos_scale
        tmp2 = torch.sum((1 - labels_true) * F.relu(self.neg_margin + logits)) * self.neg_scale
        return torch.mean(torch.add(tmp1, tmp2))


def onehot_encode(type_ids, n_types):
    tmp = np.zeros(n_types)
    for t in type_ids:
        tmp[t] = 1.0
    return tmp
