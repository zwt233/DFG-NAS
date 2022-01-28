import sys
import time
import random
import argparse
import collections
import numpy as np

import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from ogb.nodeproppred import PygNodePropPredDataset, Evaluator

from utils import *
from train import *
from operation import *
from mutation import *

parser = argparse.ArgumentParser("ogbn")
parser.add_argument('--seed', type=int, default=2, help='random seed')
parser.add_argument('--gpu', type=int, default=1, help='gpu device id')
parser.add_argument('--hiddim', type=int, default=512, help='hidden dims')
parser.add_argument('--fdrop', type=float, default=0.3, help='drop for pubmed feature')
parser.add_argument('--drop', type=float, default=0.5, help='drop for pubmed layers')
parser.add_argument('--learning_rate', type=float, default=0.001, help='init pubmed learning rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
parser.add_argument('--epochs', type=int, default=500, help='num of training epochs')
parser.add_argument('--evals', type=int, default=10, help='num of evals')
parser.add_argument('--startLength', type=int, default=4, help='num of startArch')
args = parser.parse_args()

dataset = PygNodePropPredDataset(name='ogbn-arxiv', root='../data')

r = 0.5
data = dataset[0]
N = data.num_nodes
print('Making the graph undirected.')
data.edge_index = to_undirected(data.edge_index, data.num_nodes)
row, col = data.edge_index
print('Computing adj...')
adj = SparseTensor(row=row, col=col, sparse_sizes=(N, N))
adj = adj.set_diag()
deg = adj.sum(dim=1).to(torch.float)
deg_inv_sqrt_left = deg.pow(r-1)
deg_inv_sqrt_right = deg.pow(-r)
deg_inv_sqrt_left[deg_inv_sqrt_left == float('inf')] = 0
deg_inv_sqrt_right[deg_inv_sqrt_right == float('inf')] = 0
adj = deg_inv_sqrt_left.view(-1, 1) * adj * deg_inv_sqrt_right.view(1, -1)
torch.save(adj, './arxiv_adj.pt')

adj = adj.cuda()
features = data.x
features = features.cuda()
labels = torch.reshape(data.y, (-1,))
labels = labels.cuda()
data = adj, features, labels

split_idx = dataset.get_idx_split()
idx_train, idx_val, idx_test = split_idx['train'], split_idx['valid'], split_idx['test']
idx_train = idx_train.cuda()
idx_val = idx_val.cuda()
idx_test = idx_test.cuda()
index = idx_train, idx_val, idx_test

class Model(object):
  """A class representing a model."""
  def __init__(self):
    self.arch = None
    self.val_acc = None
    self.test_acc = None
    
  def __str__(self):
    """Prints a readable version of this bitstring."""
    return self.arch

def main(cycles, population_size, sample_size):
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)
    
    """Algorithm for regularized evolution (i.e. aging evolution)."""
    population = collections.deque()
    history = []  # Not used by the algorithm, only used to report results.
    
    # Initialize the population with random models.
    while len(population) < population_size:
        model = Model()
        model.arch = random_architecture(args.startLength)
        model.val_acc, model.test_acc = train_and_eval(args, model.arch, data, index)
        population.append(model)
        history.append(model)
        print(model.arch)
        print(model.val_acc, model.test_acc)
    
    # Carry out evolution in cycles. Each cycle produces a model and removes another.
    while len(history) < cycles:
        # Sample randomly chosen models from the current population.
        sample = []
        while len(sample) < sample_size:
            candidate = random.choice(list(population))
            sample.append(candidate)
        
        # The parent is the best model in the sample.
        parent = max(sample, key=lambda i: i.val_acc)
        
        # Create the child model and store it.
        child = Model()
        child.arch = mutate_arch(parent.arch, np.random.randint(0, 3))
        child.val_acc, child.test_acc = train_and_eval(args, child.arch, data, index)
        population.append(child)
        history.append(child)
        print(child.arch)
        print(child.val_acc, child.test_acc)
        
        # Remove the oldest model.
        population.popleft()
    
    return history

# store the search history
h = main(500, 20, 3)
