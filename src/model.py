from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_max_pool, global_mean_pool


class ResidualRGCN(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hid_dim: int,
        num_relations: int,
        layers: int = 3,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lin_in = nn.Embedding(in_dim, hid_dim)
        self.gnns = nn.ModuleList(
            [
                RGCNConv(
                    hid_dim,
                    hid_dim,
                    num_relations=num_relations,
                    num_bases=min(8, num_relations),
                )
                for _ in range(layers)
            ]
        )
        self.bns = nn.ModuleList([nn.BatchNorm1d(hid_dim) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x_ids, edge_index, edge_type):
        x = self.lin_in(x_ids)  # [N, H]
        for conv, bn in zip(self.gnns, self.bns):
            h = conv(x, edge_index, edge_type)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)
            x = x + h  # residual
        return x


class GNNClassifier(nn.Module):
    """
    Relational GNN with residual+BN, mean+max readout + MLP head.
    Compatible with your trainer; adds calibrated logits (temperature settable).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        layers: int = 3,
        dropout: float = 0.3,
        num_relations: int = 4,  # AST, seq, sibling, same-ident
    ):
        super().__init__()
        self.encoder = ResidualRGCN(
            vocab_size, hidden_dim, num_relations, layers, dropout
        )
        self.lin1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin_out = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        # temperature for calibration (learned post-hoc)
        self.register_buffer("temperature", torch.ones(1))

    def forward(self, data):
        x_ids, edge_index, batch = data.x, data.edge_index, data.batch
        edge_type = getattr(data, "edge_type", None)
        if edge_type is None:
            # fallback: treat all edges as relation 0
            edge_type = torch.zeros(
                edge_index.size(1), dtype=torch.long, device=edge_index.device
            )
        x = self.encoder(x_ids, edge_index, edge_type)

        # mean + max pooling concatenation
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        g = torch.cat([x_mean, x_max], dim=1)

        g = F.relu(self.lin1(g))
        g = self.dropout(g)
        logits = self.lin_out(g)
        # temperature scaling at inference-time
        return logits / self.temperature.clamp(min=1e-4)

    @torch.no_grad()
    def set_temperature(self, val_loader, device):
        """
        Post-hoc calibration via temperature scaling on validation set (ECE-style).
        """
        self.eval()
        logits_list, labels_list = [], []
        for batch in val_loader:
            batch = batch.to(device)
            logits = super(GNNClassifier, self).forward(batch)  # unscaled
            logits_list.append(logits.detach().cpu())
            labels_list.append(batch.y.detach().cpu())
        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        # simple 1D search for best temperature (could do LBFGS)
        temps = torch.logspace(-1, 1.0, steps=30)  # 0.1 .. 10
        best_ece, best_t = float("inf"), 1.0
        for t in temps:
            probs = F.softmax(logits / t, dim=1)[:, 1]
            ece = _ece_binary(probs, labels)
            if ece < best_ece:
                best_ece, best_t = ece, float(t)
        self.temperature[...] = best_t


def _ece_binary(probs: torch.Tensor, labels: torch.Tensor, bins: int = 15) -> float:
    """Expected Calibration Error for binary probs (pos class)."""
    probs = probs.float()
    labels = labels.float()
    edges = torch.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (
            (probs >= lo) & (probs < hi)
            if i < bins - 1
            else (probs >= lo) & (probs <= hi)
        )
        if mask.any():
            conf = probs[mask].mean()
            acc = (probs[mask] >= 0.5).float().eq(labels[mask]).float().mean()
            ece += (mask.float().mean() * (acc - conf).abs()).item()
    return float(ece)
