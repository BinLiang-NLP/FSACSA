# -*- coding: utf-8 -*-
# file: BERT_SPC.py
# author: songyouwei <youwei0314@gmail.com>
# Copyright (C) 2019. All Rights Reserved.
import torch
import torch.nn as nn
from layers.induction_layer import Induction
from layers.relation import Relation


class BERT_ASPECT(nn.Module):
    def __init__(self, bert, opt):
        super(BERT_ASPECT, self).__init__()
        self.bert = bert
        opt.feature_dim = opt.bert_dim
        self.induction = Induction(opt)
        self.relation = Relation(opt)

    def forward(self, x):
        sentence,masked_sentence,seed_words,bert_segments_ids,supportset_size = x
        num_support = supportset_size.item()
        _, sentence_support_encoder = self.bert(sentence[:num_support],token_type_ids=bert_segments_ids[:num_support])
        _, sentence_query_encoder = self.bert(sentence[num_support:],token_type_ids=bert_segments_ids[num_support:])  # (k*c, 2*hidden_size)
        # _, masked_support_encoder = self.bert(masked_sentence[:num_support],token_type_ids=bert_segments_ids[:num_support])  # (k*c, 2*hidden_size)
        # _, masked_query_encoder = self.bert(masked_sentence[num_support:],token_type_ids=bert_segments_ids[num_support:])
        # _, seed_support_encoder = self.bert(seed_words[:num_support],token_type_ids=bert_segments_ids[:num_support])
        # _, seed_query_encoder =  self.bert(seed_words[num_support:],token_type_ids=bert_segments_ids[num_support:]) # (k*c, 2*hidden_size)
        ##get seed enhanced support vector
        # diff_support_vector = torch.abs(sentence_support_encoder - masked_support_encoder)
        # seed_enhanced_support_vector = diff_support_vector * seed_support_encoder
        ##get seed enhanced query vector
        # diff_query_vector = torch.abs(sentence_query_encoder - masked_query_encoder)
        # seed_enhanced_query_vector = diff_query_vector * seed_query_encoder

        # support_encoder = torch.cat((sentence_support_encoder,seed_enhanced_support_vector),dim=1)
        # query_encoder = torch.cat((sentence_query_encoder,seed_enhanced_query_vector),dim=1)

        support_encoder = sentence_support_encoder
        query_encoder = sentence_query_encoder

        class_vector = self.induction(support_encoder)
        probs = self.relation(class_vector, query_encoder)
        return probs
