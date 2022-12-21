from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
import torch.nn as nn
from enum import Enum

import logging
logger = logging.getLogger(__name__)


class BertBaseModelType(str, Enum):
    def __str__(self):
        return str(self.value)
    bert = "bert"
    char_bert = "char_bert"


class BertForSentence(BertPreTrainedModel):

    def __init__(self, config, classifier_cls):

        super(BertForSentence, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = classifier_cls(
            config.hidden_size,
            config.num_labels
        )
        self.init_weights()

    def forward(
            self,
            input_ids=None, attention_mask=None, token_type_ids=None,
            labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits_p = self.classifier(pooled_output)

        outputs = (logits_p,) + outputs[2:]
        return outputs
