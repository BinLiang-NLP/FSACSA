import torch
from torch import nn
from torch.nn.modules.loss import _Loss
from sklearn.metrics import f1_score

class Criterion(_Loss):
    def __init__(self, opt):
        way = opt.ways
        shot = opt.shots
        super(Criterion, self).__init__()
        self.amount = way * shot

    def forward(self, probs, target,num_support=None):  # (Q,C) (Q)
        if num_support is None:
            num_support = self.amount
        target = target[num_support:]
        target_onehot = torch.zeros_like(probs)
        target_onehot = target_onehot.scatter(1, target.reshape(-1, 1), 1)
        loss = torch.mean((probs - target_onehot) ** 2)
        pred = torch.argmax(probs, dim=1)
        assert target.shape[0] == pred.shape[0], "target len != pred len"
        acc = torch.sum(target == pred).float() / target.shape[0]
        f1 = f1_score(target.data.cpu().numpy(), pred.data.cpu().numpy(), average='macro')
        return pred, loss, acc, f1


class CrossEntropyCriterion(_Loss):
    def __init__(self, opt):
        way = opt.ways
        shot = opt.shots
        super(CrossEntropyCriterion, self).__init__()
        self.amount = way * shot
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, probs, target,num_support=None):  # (Q,C) (Q)
        if num_support is None:
            num_support = self.amount
        target = target[num_support:]
        loss = self.ce_loss(target,probs)
        pred = torch.argmax(probs, dim=1)
        assert target.shape[0] == pred.shape[0], "target len != pred len"
        acc = torch.sum(target == pred).float() / target.shape[0]
        f1 = f1_score(target.data.cpu().numpy(), pred.data.cpu().numpy(), average='macro')
        return pred, loss, acc, f1
