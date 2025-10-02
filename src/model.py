# src/model.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_max_pool, global_mean_pool


class ResidualRGCN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        feat_dim: int,
        hidden_dim: int,
        num_relations: int,
        layers: int = 3,
        dropout: float = 0.3,
        num_bases: int = 8,
        use_virtual_node: bool = True,
    ):
        super().__init__()
        self.emb: nn.Embedding = nn.Embedding(vocab_size, hidden_dim)
        self.feat_mlp: nn.Sequential = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
        )
        self.use_vn: bool = use_virtual_node
        if self.use_vn:
            self.virtual: nn.Parameter = nn.Parameter(torch.zeros(1, hidden_dim))

        self.convs: nn.ModuleList = nn.ModuleList(
            [
                RGCNConv(
                    hidden_dim,
                    hidden_dim,
                    num_relations=num_relations,
                    num_bases=min(num_bases, num_relations),
                )
                for _ in range(layers)
            ]
        )
        self.bns: nn.ModuleList = nn.ModuleList(
            [nn.BatchNorm1d(hidden_dim) for _ in range(layers)]
        )
        self.dropout: nn.Dropout = nn.Dropout(dropout)

    def forward(self, data) -> torch.Tensor:
        x_ids: torch.Tensor = data.x
        edge_index: torch.Tensor = data.edge_index
        edge_type: Optional[torch.Tensor] = getattr(data, "edge_type", None)
        if edge_type is None:
            edge_type = torch.zeros(
                edge_index.size(1), dtype=torch.long, device=edge_index.device
            )
        node_feat: Optional[torch.Tensor] = getattr(data, "node_feat", None)
        if node_feat is None:
            node_feat = torch.zeros((x_ids.size(0), 6), device=x_ids.device)

        x = self.emb(x_ids) + self.feat_mlp(node_feat)

        if self.use_vn:
            # broadcast virtual node per-graph and add
            vn = self.virtual.expand(x.size(0), -1)
            x = x + vn

        for conv, bn in zip(self.convs, self.bns):
            h = conv(x, edge_index, edge_type)
            h = bn(h)
            h = F.relu(h)
            h = self.dropout(h)
            x = x + h  # residual

        return x


class GNNClassifier(nn.Module):
    """
    Relational GNN with residuals+BN, virtual node, mean+max readout, MLP head,
    temperature scaling for calibration.
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_dim: int = 128,
        num_classes: int = 2,
        layers: int = 3,
        dropout: float = 0.3,
        num_relations: int = 4,  # AST, seq, sibling, same-ident
        node_feat_dim: int = 6,  # [is_id,is_num,is_str,in_cond,deg_in,deg_out]
    ):
        super().__init__()
        self.encoder: ResidualRGCN = ResidualRGCN(
            vocab_size=vocab_size,
            feat_dim=node_feat_dim,
            hidden_dim=hidden_dim,
            num_relations=num_relations,
            layers=layers,
            dropout=dropout,
        )
        self.lin1: nn.Linear = nn.Linear(hidden_dim * 2, hidden_dim)
        self.lin_out: nn.Linear = nn.Linear(hidden_dim, num_classes)
        self.dropout: nn.Dropout = nn.Dropout(dropout)
        # temperature buffer for post-hoc calibration
        self.register_buffer("temperature", torch.ones(1))

    # ---- internal helper: unscaled logits (no temperature applied) ----
    def _logits_unscaled(self, data) -> torch.Tensor:
        # call .forward explicitly (pyright-safe)
        x = self.encoder.forward(data)
        x_mean = global_mean_pool(x, data.batch)
        x_max = global_max_pool(x, data.batch)
        g = torch.cat([x_mean, x_max], dim=1)
        g = F.relu(self.lin1(g))
        g = self.dropout(g)
        return self.lin_out(g)

    def forward(self, data) -> torch.Tensor:
        logits = self._logits_unscaled(data)
        # apply temperature scaling safely
        return logits / self.temperature.clamp(min=1e-4)

    @torch.no_grad()
    def set_temperature(self, val_loader, device) -> None:
        """
        Post-hoc temperature scaling fitted by minimizing ECE over a grid.
        """
        self.eval()
        logits_list, labels_list = [], []
        for batch in val_loader:
            batch = batch.to(device)
            logits = self._logits_unscaled(batch)  # UNscaled
            logits_list.append(logits.detach().cpu())
            labels_list.append(batch.y.detach().cpu())
        logits = torch.cat(logits_list, dim=0)
        labels = torch.cat(labels_list, dim=0)

        temps = torch.logspace(-1, 1.0, steps=30)  # 0.1 .. 10
        best_ece, best_t = float("inf"), 1.0
        for t in temps:
            probs = F.softmax(logits / t, dim=1)[:, 1]
            ece = _ece_binary(probs, labels)
            if ece < best_ece:
                best_ece, best_t = ece, float(t)
        # write to buffer without indexing (pyright-safe)
        self.temperature.fill_(best_t)


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
