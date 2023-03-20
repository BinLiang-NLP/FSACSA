import torch
import torch.nn as nn
from layers.lstm_encoder import LSTMEncoder,ATAE_LSTMEncoder
from layers.induction_layer import Induction
from layers.relation import Relation





class FewShotInduction(nn.Module):
    def __init__(self, embedding_matrix,opt):
        super(FewShotInduction, self).__init__()
        self.encoder = LSTMEncoder(opt,embedding_matrix)
        opt.feature_dim = 2*opt.hidden_dim
        self.induction = Induction(opt)
        self.relation = Relation(opt)

    def forward(self, x):
        num_support = x[1].item()
        input_x=x[0]
        support_encoder, query_encoder = self.encoder(input_x,num_support = num_support)  # (k*c, 2*hidden_size)
        class_vector = self.induction(support_encoder)
        probs = self.relation(class_vector, query_encoder)
        return probs


class AspectAwareInduction(nn.Module):
    def __init__(self, embedding_matrix,opt):
        super(AspectAwareInduction, self).__init__()
        self.encoder = ATAE_LSTMEncoder(embedding_matrix,opt)
        opt.feature_dim = opt.hidden_dim
        self.induction = Induction(opt)
        self.relation = Relation(opt)

    def forward(self, x):
        num_support = x[2].item()
        input_x=x[0]
        aspect_indices = x[1]
        support_input_x = input_x[:num_support]
        query_input_x = input_x[num_support:]
        support_aspect_indices = aspect_indices[:num_support]
        query_aspect_indices = aspect_indices[num_support:]
        support_encoder = self.encoder(support_input_x,support_aspect_indices)  # (k*c, 2*hidden_size)
        query_encoder = self.encoder(query_input_x,query_aspect_indices)  # (k*c, 2*hidden_size)
        class_vector = self.induction(support_encoder)
        probs = self.relation(class_vector, query_encoder)
        return probs




