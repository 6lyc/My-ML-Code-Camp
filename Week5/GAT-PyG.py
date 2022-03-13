'''
base Pyg
'''
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid

dataset = Planetoid(root='./tmp/Cora', name='Cora')

model_config = {
        'layers': [100, 50, 10],
        'num_head': 4,
        'is_self_loop': True,
        'dropout_rate': 0.3,
    }

train_config = {
        'learning_rate': 0.01,
        'weight_decay': 0.00001,
}

class GAT_NET(torch.nn.Module):
    def __init__(self, features, hidden, classes, heads=4):
        super(GAT_NET, self).__init__()
        self.gat1 = GATConv(features, hidden, heads=4)  # 定义GAT层，使用多头注意力机制
        self.gat2 = GATConv(hidden*heads, classes)  # 因为多头注意力是将向量拼接，所以维度乘以头数。

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.gat1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.gat2(x, edge_index)

        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = dataset[0].to(device)

A = []
for i in range(10):
    model = GAT_NET(dataset.num_node_features, 16, dataset.num_classes, heads=4).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    model.train()
    for epoch in range(200):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()

    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    A.append(acc)
    print(f'Accuracy: {acc:.4f}')

print(np.mean(A))