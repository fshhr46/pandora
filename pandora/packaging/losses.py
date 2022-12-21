
from enum import Enum

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import CrossEntropyLoss


class LossType(str, Enum):
    def __str__(self):
        return str(self.value)
    x_ent = "x_ent"
    focal_loss = "focal_loss"

    @classmethod
    def get_loss_func(cls, loss_type, device, config={}):
        if loss_type == cls.x_ent:
            loss_func = CrossEntropyLoss(**config)
        elif loss_type == cls.focal_loss:
            data_distributions = config.get("data_distributions", {})
            # Converting
            if data_distributions and data_distributions["train"]:
                data_distribution = data_distributions["train"]
                label2id = config["label2id"]

                label2id_list = sorted(label2id.items(), key=lambda tp: tp[1])
                alphas = []
                for label2id in label2id_list:
                    label = label2id[0]
                    # use inverse frequency of the label as alpha
                    alpha_t = 1 - data_distribution[label]
                    alphas.append(alpha_t)
                alphas = torch.tensor(alphas).to(device)
            else:
                alphas = None
            loss_func = FocalLoss(alphas=alphas)
        else:
            raise ValueError(f"unknown loss_type {loss_type}")
        return loss_func


class FocalLoss(nn.Module):
    '''Multi-class Focal loss implementation'''

    def __init__(self, gamma=2, alphas=None, reduction='mean', ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alphas = alphas
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        ce_loss = F.cross_entropy(
            input, target, reduction=self.reduction, weight=self.alphas)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma * ce_loss).mean()
        return focal_loss

    # Deprecated
    def forward_old(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        logpt = F.log_softmax(input, dim=1)
        pt = torch.exp(logpt)
        logpt = (1-pt)**self.gamma * logpt
        loss = F.nll_loss(logpt, target, self.alphas,
                          ignore_index=self.ignore_index)
        return loss
