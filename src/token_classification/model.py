#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
The current implementation has repeated code but will guarantee the performance for each model.
"""

import torch
import torch.nn.functional as F
from torch import nn
from transformers import (XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST,
                          XLMRobertaConfig, XLMRobertaModel, XLMRobertaModel, XLMRobertaPreTrainedModel)

from token_classification.model_utils import FocalLoss, _calculate_loss
from token_classification.model_utils import New_Transformer_CRF as Transformer_CRF


class XLMRobertaNerModel(XLMRobertaPreTrainedModel):
    config_class = XLMRobertaConfig
    pretrained_model_archive_map = XLM_ROBERTA_PRETRAINED_MODEL_ARCHIVE_LIST
    base_model_prefix = "roberta"

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.roberta = XLMRobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.loss_fct = nn.CrossEntropyLoss()
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, label_ids=None):
        """
        :return: raw logits without any softmax or log_softmax transformation

        qoute for reason (https://discuss.pytorch.org/t/logsoftmax-vs-softmax/21386/7):
        You should pass raw logits to nn.CrossEntropyLoss, since the function itself applies F.log_softmax and nn.NLLLoss() on the input.
        If you pass log probabilities (from nn.LogSoftmax) or probabilities (from nn.Softmax()) your loss function won’t work as intended.

        From the pytorch CrossEntropyLoss doc:
        The input is expected to contain raw, unnormalized scores for each class.
        """
        outputs = self.roberta(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)

        seq_outputs = outputs[0]
        seq_outputs = self.dropout(seq_outputs)
        logits = self.classifier(seq_outputs)

        loss, active_logits = _calculate_loss(logits, attention_mask, label_ids, self.loss_fct, self.num_labels)

        return logits, active_logits, loss