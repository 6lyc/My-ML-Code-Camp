import random
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.datasets import Planetoid
import numpy as np

dataset = Planetoid(root='/tmp/Cora', name='Cora')

class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset[0].to(device)

train_ps = 0.6
id = list(range(data.num_nodes))
random.shuffle(id)
# print(id)
train_idx = id[0:int(data.num_nodes*train_ps)]
test_idx = id[int(data.num_nodes*train_ps):data.num_nodes]
A = []
for i in range(10):
    model = GCN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[train_idx], data.y[train_idx])
        loss.backward()
        optimizer.step()

    # print(data.train_mask)
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[test_idx] == data.y[test_idx]).sum()
    acc = int(correct) / int(data.num_nodes*(1 - train_ps))
    A.append(acc)
    print(f'Accuracy: {acc:.4f}')

print(np.mean(A))

