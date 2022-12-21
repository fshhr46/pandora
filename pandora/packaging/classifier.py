
import torch.nn as nn
from torch import Tensor
from torch.nn import Module
from enum import Enum

import logging
logger = logging.getLogger(__name__)


class LinearClassifier(Module):
    def __init__(self,
                 hidden_size: int,
                 num_labels: int) -> None:
        super(LinearClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input: Tensor) -> Tensor:
        return self.classifier(input)


class DOCClassifier(Module):
    def __init__(self,
                 hidden_size: int,
                 num_labels: int,
                 threshold: float = 0.5) -> None:
        """ doc_config: This is used for training the model to detect Unseen classes. 
            This comes from the paper: DOC: Deep Open Classification of Text Documents
            See https://arxiv.org/abs/1709.08716
        """
        super(DOCClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.threshold = threshold
        self.sigmoids = {i: nn.Sigmoid() for i in range(num_labels)}

    def forward(self, input: Tensor) -> Tensor:
        logits_list = []
        for _, sigmoid_func in self.sigmoids.items():
            logits_list.append(sigmoid_func(input))


class ClassifierType(str, Enum):
    def __str__(self):
        return str(self.value)
    linear = "linear"
    # deep_open_classifier: a classifier with unknown output.
    doc = "deep_open_classifier"

    @classmethod
    def get_classifier_cls(cls, classifier_type):
        if classifier_type == cls.linear:
            classifier_cls = nn.Linear
        elif classifier_type == cls.doc:
            classifier_cls = DOCClassifier
        else:
            raise ValueError(f"unknown classifier_type {classifier_type}")
        return classifier_cls
