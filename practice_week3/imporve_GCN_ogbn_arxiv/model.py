import dgl
import dgl.nn
import torch.nn as nn
import torch.nn.functional as F

class StochasticTwoLayerGCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.conv1 = dgl.nn.GraphConv(in_features, hidden_features)
        self.conv2 = dgl.nn.GraphConv(hidden_features, out_features)
        self.BN1 = nn.BatchNorm1d(hidden_features)
        self.BN2 = nn.BatchNorm1d(out_features)

    def forward(self, blocks, x):
        x = F.relu(self.BN1(self.conv1(blocks[0], x)))
        x = F.dropout(x, 0.5)
        x = F.relu(self.BN2(self.conv2(blocks[1], x)))
        return x