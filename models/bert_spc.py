# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn


class BERT_SPC(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_SPC, self).__init__()
        self.bert = bert
        self.dropout = nn.Dropout(opt.dropout)
        self.dense = nn.Linear(opt.bert_dim, opt.ways)

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids,supportset_size = inputs[0], inputs[1],inputs[2]
        supportset_size = supportset_size.item()
        text_bert_indices = text_bert_indices[supportset_size:]
        bert_segments_ids = bert_segments_ids[supportset_size:]
        _, pooled_output = self.bert(text_bert_indices, token_type_ids=bert_segments_ids)
        pooled_output = self.dropout(pooled_output)
        logits = self.dense(pooled_output)
        return logits
