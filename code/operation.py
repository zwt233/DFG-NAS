import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

class Graph(nn.Module):
    def __init__(self, adj):
        super(Graph, self).__init__()
        self.adj = adj

    def forward(self, x):
        x = self.adj.matmul(x)
        return x

class MLP(nn.Module):
    def __init__(self, nfeat, nclass, dropout, last=False):
        super(MLP, self).__init__()
        self.lr1 = nn.Linear(nfeat, nclass)
        self.dropout = dropout
        self.last = last

    def forward(self, x):
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lr1(x)
        if not self.last:
            x = F.relu(x)
        return x

class ModelOp(nn.Module):
    def __init__(self, arch, adj, feat_dim, hid_dim, num_classes, fdropout, dropout):
        super(ModelOp, self).__init__()
        self._ops = nn.ModuleList()
        self._numP = 1
        self._arch = arch
        for element in arch:
            if element == 1:
                op = Graph(adj)
                self._numP += 1
            elif element == 0:
                op = MLP(hid_dim, hid_dim, dropout)
            else:
                print("arch element error")
            self._ops.append(op)
        self.gate = torch.nn.Parameter(1e-5*torch.randn(self._numP), requires_grad=True)
        self.preprocess0 = MLP(feat_dim, hid_dim, fdropout, True)
        self.classifier = MLP(hid_dim, num_classes, dropout, True)
    
    def forward(self, s0):
        tempP, numP, point, totalP = [], [], 0, 0
        tempT  = []
        for i in range(len(self._arch)):
            if i == 0:
                res = self.preprocess0(s0)
                tempP.append(res)
                numP.append(i)
                totalP += 1
                
                res = self._ops[i](res)
                if self._arch[i] == 1:
                    tempP.append(res)
                    numP.append(i)
                    totalP += 1
                else:
                    tempT.append(res)
                    numP = []
                    tempP = []
            else:
                if self._arch[i - 1] == 1:
                    if self._arch[i] == 0:
                        res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
                        res = self._ops[i](res)
                        tempT.append(res)
                        numP = []
                        tempP = []
                    else:
                        res = self._ops[i](res)
                        tempP.append(res)
                        numP.append(i - point)
                        totalP += 1
                else:
                    if self._arch[i] == 1:
                        res = self._ops[i](res)
                        tempP.append(res)
                        point = i
                        numP.append(i - point)
                        totalP += 1
                    else:
                        res = sum(tempT)
                        res = self._ops[i](res)
                        tempT.append(res)
        if len(numP) > 0 or len(tempP) > 0:
            res = sum([torch.mul(F.sigmoid(self.gate[totalP - len(numP) + i]), tempP[i]) for i in numP])
        else:
            res = sum(tempT)
        res = self.classifier(res)
        logits = F.log_softmax(res, dim=1)
        return logits