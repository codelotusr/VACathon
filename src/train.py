import argparse
import json
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .model import GNNClassifier


def load_graphs(path: str) -> list[Data]:
    torch.serialization.add_safe_globals([Data])
    p = Path(path)
    if p.exists():
        return torch.load(p, weights_only=False)
    shards = sorted(p.parent.glob(p.stem + ".*.pt"))
    graphs: list[Data] = []
    for shard in shards:
        graphs.extend(torch.load(shard, weights_only=False))
    if not graphs:
        raise FileNotFoundError(f"No graphs found for {path} (or shards).")
    return graphs


def load_data(data_dir="data/processed", batch_size=32):
    train_graphs = load_graphs(f"{data_dir}/train.pt")
    val_graphs = load_graphs(f"{data_dir}/validation.pt")
    test_graphs = load_graphs(f"{data_dir}/test.pt")

    return (
        DataLoader(train_graphs, batch_size=batch_size, shuffle=True),
        DataLoader(val_graphs, batch_size=batch_size),
        DataLoader(test_graphs, batch_size=batch_size),
    )


def _class_weights(loader) -> torch.Tensor:
    # compute per-class counts
    import numpy as np

    counts = np.zeros(2, dtype=float)
    for batch in loader:
        y = batch.y.numpy()
        for c in (0, 1):
            counts[c] += (y == c).sum()
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    w = inv / inv.sum() * 2.0
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: torch.Tensor | float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        if isinstance(self.alpha, torch.Tensor):
            at = self.alpha.to(logits.device)[targets]
        else:
            at = self.alpha
        loss = at * ((1 - pt) ** self.gamma) * ce
        return loss.mean()


@torch.no_grad()
def eval_full(model, loader, device) -> Tuple[float, float, float, float, float]:
    """Returns: acc, precision, recall, f1, ece"""
    model.eval()
    correct, total = 0, 0
    all_probs, all_labels = [], []
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch)
        probs = F.softmax(logits, dim=1)[:, 1]
        pred = probs >= 0.5
        correct += int(pred.eq(batch.y.bool()).sum())
        total += batch.num_graphs
        all_probs.append(probs.detach().cpu())
        all_labels.append(batch.y.detach().cpu())
    import torch as T

    probs = T.cat(all_probs)
    labels = T.cat(all_labels)
    acc = correct / total if total > 0 else 0.0
    from .model import _ece_binary

    ece = _ece_binary(probs, labels)
    pr, rc, f1, _ = precision_recall_fscore_support(
        labels.numpy(),
        (probs.numpy() >= 0.5).astype(int),
        average="binary",
        zero_division=0,
    )
    return acc, pr, rc, f1, ece


def train_model(
    epochs=20,
    lr=1e-3,
    hidden_dim=128,
    batch_size=32,
    data_dir="data/processed",
    ckpt_path="checkpoints/model.pt",
    patience=10,
    use_focal=True,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = json.load(open(f"{data_dir}/node_vocab.json"))
    model = GNNClassifier(len(vocab), hidden_dim=hidden_dim).to(device)

    train_loader, val_loader, test_loader = load_data(data_dir, batch_size)

    # Class weights + focal (helps imbalanced slices like in BigVul)  :contentReference[oaicite:3]{index=3}
    w = _class_weights(train_loader).to(device)
    if use_focal:
        criterion = FocalLoss(alpha=w, gamma=2.0)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=w)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )

    best_val_f1 = -1.0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss.item())

        # Eval
        val_acc, val_pr, val_rc, val_f1, val_ece = eval_full(model, val_loader, device)
        scheduler.step(val_f1)

        print(
            f"Epoch {epoch:03d} | Loss {total_loss/ max(1,len(train_loader)):.4f} | "
            f"Val Acc {val_acc:.4f} | P {val_pr:.4f} | R {val_rc:.4f} | F1 {val_f1:.4f} | ECE {val_ece:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            stale = 0
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ Saved new best model → {ckpt_path} (val_f1={best_val_f1:.4f})")
        else:
            stale += 1
            if stale >= patience:
                print(
                    f"⏹️  Early stopping: no val F1 improvement for {patience} epochs."
                )
                break

    # Load best + calibrate temperature on validation
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.set_temperature(val_loader, device)

    test_acc, test_pr, test_rc, test_f1, test_ece = eval_full(
        model, test_loader, device
    )
    print("\nFinal test:")
    print(
        f"  Acc {test_acc:.4f} | P {test_pr:.4f} | R {test_rc:.4f} | F1 {test_f1:.4f} | ECE {test_ece:.4f}"
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train relational GNN on code vulnerability dataset"
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--ckpt-path", type=str, default="checkpoints/model.pt")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--no-focal", action="store_true")
    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        ckpt_path=args.ckpt_path,
        patience=args.patience,
        use_focal=not args.no_focal,
    )
