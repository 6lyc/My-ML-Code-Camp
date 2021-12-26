import dgl
import torch
from ogb.nodeproppred import DglNodePropPredDataset


def getdata():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = DglNodePropPredDataset('ogbn-arxiv')
    g, node_label = dataset[0]
    g = dgl.add_reverse_edges(g)
    g.ndata['label'] = node_label[:, 0]

    # add self loop(one time)
    g = dgl.remove_self_loop(g)
    g = dgl.add_self_loop(g)

    # normalization
    degs = g.in_degrees().float()
    norm = torch.pow(degs, -0.5)
    norm[torch.isinf(norm)] = 0
    g.ndata['norm'] = norm.unsqueeze(1)

    idx_split = dataset.get_idx_split()
    train_nids = idx_split['train']
    valid_nids = idx_split['valid']
    test_nids = idx_split['test']

    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)
    train_dataloader = dgl.dataloading.NodeDataLoader(
        g, train_nids, sampler,
        batch_size=1024,
        device=device,
        shuffle=True,
        drop_last=False,
        num_workers=0)

    valid_dataloader = dgl.dataloading.NodeDataLoader(
        g, valid_nids, sampler,
        device=device,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )

    test_dataloader = dgl.dataloading.NodeDataLoader(
        g, test_nids, sampler,
        device=device,
        batch_size=1024,
        shuffle=False,
        drop_last=False,
        num_workers=0
    )
    return g, train_dataloader, valid_dataloader, test_dataloader, node_label