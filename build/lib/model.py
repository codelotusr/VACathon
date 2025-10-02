import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_mean_pool


class GNNClassifier(nn.Module):
    def __init__(self, vocab_size, hidden_dim=64, num_classes=2, steps=3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.ggnn = GatedGraphConv(hidden_dim, num_layers=steps)
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.embedding(x)
        x = self.ggnn(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        return self.lin_out(x)
