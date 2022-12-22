
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
    def get_loss_func(cls, loss_type, device, use_doc, config={}):
        if loss_type == cls.x_ent:
            loss_func = CrossEntropyLoss(use_doc, **config)
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
            loss_func = FocalLoss(use_doc, alphas=alphas)
        else:
            raise ValueError(f"unknown loss_type {loss_type}")
        return loss_func


class LossFuncBase(nn.Module):
    def __init__(self, use_doc, *args, **kwargs):
        super(LossFuncBase, self).__init__(*args, **kwargs)
        self.use_doc = use_doc

    def calculate_doc_loss(self, sigmoids, target, reduction='sum'):
        """
        N: Batch size.
        C: Number of categories (num_labels)
        logits: [N, C]
        target: [N, ]
        """
        num_labels = sigmoids.shape[1]
        target_one_hot = F.one_hot(
            target, num_classes=num_labels).to(torch.float32)

        return F.binary_cross_entropy(
            sigmoids,
            target_one_hot,
            reduction=reduction)

    def calculate_xent(self, logits, target, weight=None):
        ce_loss = F.cross_entropy(
            logits, target, weight=weight)
        return ce_loss


class CrossEntropyLoss(LossFuncBase):
    def __init__(self, use_doc, *args, **kwargs):
        super(CrossEntropyLoss, self).__init__(use_doc, *args, **kwargs)

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        if self.use_doc:
            return self.calculate_doc_loss(input, target)
        else:
            return self.calculate_xent(input, target)


class FocalLoss(LossFuncBase):
    '''Multi-class Focal loss implementation'''

    def __init__(self, use_doc, gamma=2, alphas=None, *args, **kwargs):
        super(FocalLoss, self).__init__(use_doc, *args, **kwargs)
        self.gamma = gamma
        self.alphas = alphas

    def forward(self, input, target):
        """
        input: [N, C]
        target: [N, ]
        """
        if self.use_doc:
            base_loss = self.calculate_doc_loss(input, target)
        else:
            base_loss = self.calculate_xent(input, target)
        pt = torch.exp(-base_loss)
        focal_loss = ((1 - pt) ** self.gamma * base_loss).mean()
        return focal_loss
