# src/model.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GatedGraphConv, global_mean_pool


class GNNClassifier(nn.Module):
    """
    Gated Graph Neural Network for code vulnerability detection.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        steps: int = 6,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_dim)

        # GGNN layer (message passing with gating)
        self.ggnn = GatedGraphConv(out_channels=hidden_dim, num_layers=steps)

        # Classifier head
        self.lin1 = nn.Linear(hidden_dim, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, data):
        """
        data: PyG Data object with attributes:
          - x: [num_nodes] (node ids -> vocab)
          - edge_index: [2, num_edges]
          - batch: [num_nodes] (graph ids for batching)
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Embedding → GGNN → pooling → classifier
        x = self.embedding(x)
        x = self.ggnn(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        # Pool to graph-level representation
        x = global_mean_pool(x, batch)

        x = F.relu(self.lin1(x))
        x = self.dropout(x)
        return self.lin_out(x)
