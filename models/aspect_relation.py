import torch
import torch.nn as nn
from layers.lstm_encoder import LSTMEncoder
from layers.induction_layer import Induction
from layers.relation_compare import RelationCompare


class AspectRelation(nn.Module):
    def __init__(self, embedding_matrix,opt):
        super(AspectRelation, self).__init__()
        self.encoder = LSTMEncoder(opt,embedding_matrix)
        opt.feature_dim = 4*opt.hidden_dim
        self.induction = Induction(opt)
        self.relation = RelationCompare(opt.ways,opt.feature_dim)
        self.opt = opt

    def forward(self, x):
        sentence,masked_sentence,seed_words,supportset_size = x
        num_support = supportset_size.item()
        sentence_support_encoder, sentence_query_encoder = self.encoder(sentence,num_support=num_support)  # (k*c, 2*hidden_size)
        masked_support_encoder, masked_query_encoder = self.encoder(sentence,num_support=num_support)  # (k*c, 2*hidden_size)
        seed_support_encoder, seed_query_encoder = self.encoder(sentence,num_support=num_support)  # (k*c, 2*hidden_size)
        ##get seed enhanced support vector
        diff_support_vector = torch.abs(sentence_support_encoder - masked_support_encoder)
        seed_enhanced_support_vector = diff_support_vector * seed_support_encoder
        ##get seed enhanced query vector
        diff_query_vector = torch.abs(sentence_query_encoder - masked_query_encoder)
        seed_enhanced_query_vector = diff_query_vector * seed_query_encoder

        support_encoder = torch.cat((sentence_support_encoder,seed_enhanced_support_vector),dim=1)
        # support_encoder = sentence_support_encoder+seed_enhanced_support_vector
        # query_encoder = sentence_query_encoder+seed_enhanced_query_vector
        query_encoder = torch.cat((sentence_query_encoder,seed_enhanced_query_vector),dim=1)

        class_vector = torch.mean(torch.reshape(support_encoder,
                                               (self.opt.ways, num_support//self.opt.ways, support_encoder.shape[1])), 1)
        probs = self.relation(class_vector, query_encoder)
        return probs



