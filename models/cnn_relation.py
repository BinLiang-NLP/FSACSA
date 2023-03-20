import torch
import torch.nn as nn
from layers.cnn_encoder import CNNEncoder
from layers.induction_layer import Induction
from layers.relation_compare import RelationCompare

class CNNRelation(nn.Module):
    def __init__(self, embedding_matrix,opt):
        super(CNNRelation, self).__init__()
        self.encoder = CNNEncoder(opt,embedding_matrix)
        self.opt = opt
        opt.feature_dim = 2*opt.hidden_dim
        self.induction = Induction(opt)
        self.relation = RelationCompare(opt.ways,opt.feature_dim)

    def forward(self, x):
        num_support = x[1].item()
        x = x[0]
        support_encoder, query_encoder = self.encoder(x,num_support=num_support)  # (k*c, 2*hidden_size)
        class_vector = torch.mean(torch.reshape(support_encoder,
                                               (self.opt.ways, num_support//self.opt.ways, support_encoder.shape[1])), 1)
        probs = self.relation(class_vector, query_encoder)
        return probs



