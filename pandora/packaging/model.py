import torch.nn as nn
# from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from models.transformers.modeling_bert import BertPreTrainedModel, BertModel
from torch.nn import CrossEntropyLoss

import logging
logger = logging.getLogger(__name__)


class BertForSentence(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSentence, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_type = config.loss_type
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, labels=None):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits_p = self.classifier(pooled_output)
        # TODO: only support cs for now
        # assert self.loss_type in ['lsr', 'focal', 'ce']
        # if self.loss_type == 'lsr':
        #     loss_fct = LabelSmoothingCrossEntropy()
        # elif self.loss_type == 'focal':
        #     loss_fct = FocalLoss()
        # else:
        #     loss_fct = CrossEntropyLoss()
        assert self.loss_type in ['ce']
        loss_fct = CrossEntropyLoss()

        outputs = (logits_p,) + outputs[2:]

        if labels is not None:
            loss = loss_fct(logits_p, labels)
            outputs = (loss,) + outputs
        return outputs
