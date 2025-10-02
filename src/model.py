import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

class Model(nn.Module):
    def __init__(self, vocab_size, dim=64):
        super().__init__()
        self.emb  = nn.Embedding(vocab_size, dim)
        self.gcn1 = GCNConv(dim, dim)
        self.gcn2 = GCNConv(dim, dim)
        self.cls  = nn.Linear(dim, 2)  # binary defect label

    def forward(self, data):
        x = self.emb(data.x)                  # [num_nodes, dim]
        x = self.gcn1(x, data.edge_index).relu()
        x = self.gcn2(x, data.edge_index).relu()
        x = global_mean_pool(x, data.batch)   # [batch_size, dim]
        return self.cls(x)

#model = Model(vocab_size)
