
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from enum import Enum
import torch.nn.functional as F

import logging
logger = logging.getLogger(__name__)


class LinearClassifier(Module):
    def __init__(self,
                 hidden_size: int,
                 num_labels: int) -> None:
        super(LinearClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)

    # Input tensor shape: batch_size * hidden_size
    def forward(self, input: Tensor) -> Tensor:
        logits = self.classifier(input)
        return (logits, None)


class DeepOpenClassifier(Module):
    def __init__(self,
                 hidden_size: int,
                 num_labels: int) -> None:
        """This comes from the paper: DOC: Deep Open Classification of Text Documents
            See https://arxiv.org/abs/1709.08716
        """
        super(DeepOpenClassifier, self).__init__()
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)

    # Input tensor shape: batch_size * hidden_size
    def forward(self, input: Tensor) -> Tensor:
        logits = self.classifier(input)
        dropout_out = self.dropout(logits)
        sigmoid_out = self.sigmoid(dropout_out)
        return (logits, sigmoid_out)


class ClassifierType(str, Enum):
    def __str__(self):
        return str(self.value)
    linear = "linear"
    # deep_open_classifier: a classifier with unknown output.
    doc = "deep_open_classifier"

    @classmethod
    def get_classifier_cls(cls, classifier_type):
        if classifier_type == cls.linear:
            classifier_cls = LinearClassifier
        elif classifier_type == cls.doc:
            classifier_cls = DeepOpenClassifier
        else:
            raise ValueError(f"unknown classifier_type {classifier_type}")
        return classifier_cls
