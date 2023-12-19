import torch
import dgl.data
from model import GCN
from train import train

#Loading Cora Dataset
dataset = dgl.data.CoraGraphDataset()
'''
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
'''
print('Number of categories:', dataset.num_classes)

g = dataset[0]
print('Node features')
print(g.ndata)
print('Edge features')
print(g.edata)

# add self loop(one time)
g = dgl.remove_self_loop(g)
g = dgl.add_self_loop(g)

# normalization
degs = g.in_degrees().float()
norm = torch.pow(degs, -0.5)
norm[torch.isinf(norm)] = 0
g.ndata['norm'] = norm.unsqueeze(1)

#Defining a Graph Convolutional Network (GCN)
model = GCN(g.ndata['feat'].shape[1], 16, dataset.num_classes)
#Training the GCN
train(g, model)
