# src/train.py
import argparse
import json
from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .model import GNNClassifier


def load_graphs(path: str) -> list[Data]:
    """Load one or multiple shards of graphs into a single list."""
    # Allowlist PyG Data for PyTorch 2.6+ default weights_only=True
    torch.serialization.add_safe_globals([Data])

    p = Path(path)
    if p.exists():
        return torch.load(p, weights_only=False)

    # If base file doesn't exist, try shard pattern e.g. train.pt.000.pt, train.pt.001.pt, ...
    shards = sorted(p.parent.glob(p.stem + ".*.pt"))
    graphs: list[Data] = []
    for shard in shards:
        graphs.extend(torch.load(shard, weights_only=False))
    if not graphs:
        raise FileNotFoundError(f"No graphs found for {path} (or shards).")
    return graphs


def load_data(data_dir="data/processed", batch_size=32):
    """Load train/validation/test sets into PyG DataLoaders."""
    train_graphs = load_graphs(f"{data_dir}/train.pt")
    val_graphs = load_graphs(f"{data_dir}/validation.pt")
    test_graphs = load_graphs(f"{data_dir}/test.pt")

    return (
        DataLoader(train_graphs, batch_size=batch_size, shuffle=True),
        DataLoader(val_graphs, batch_size=batch_size),
        DataLoader(test_graphs, batch_size=batch_size),
    )


def eval_model(model, loader, device) -> float:
    """Compute accuracy on a DataLoader."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
            total += batch.num_graphs
    return (correct / total) if total > 0 else 0.0


def train_model(
    epochs=20,
    lr=1e-3,
    hidden_dim=64,
    batch_size=32,
    data_dir="data/processed",
    ckpt_path="checkpoints/model.pt",
    patience=10,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load vocab size for embedding
    vocab = json.load(open(f"{data_dir}/node_vocab.json"))
    model = GNNClassifier(len(vocab), hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = load_data(data_dir, batch_size)

    best_val_acc = -1.0
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())

        avg_loss = total_loss / max(1, len(train_loader))
        val_acc = eval_model(model, val_loader, device)
        print(f"Epoch {epoch:03d} | Loss {avg_loss:.4f} | Val Acc {val_acc:.4f}")

        # Early-stopping logic (by validation accuracy)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            stale = 0
            Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), ckpt_path)
            print(
                f"  ✓ Saved new best model → {ckpt_path} (val_acc={best_val_acc:.4f})"
            )
        else:
            stale += 1
            if stale >= patience:
                print(f"⏹️  Early stopping: no val improvement for {patience} epochs.")
                break

    # Final test evaluation with best checkpoint
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    test_acc = eval_model(model, test_loader, device)
    print(f"\nFinal test accuracy (best model): {test_acc:.4f}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GGNN on code vulnerability dataset"
    )
    parser.add_argument("--epochs", type=int, default=20, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden layer size")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory with .pt files",
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default="checkpoints/model.pt",
        help="Path to save best checkpoint",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (epochs without improvement)",
    )

    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        ckpt_path=args.ckpt_path,
        patience=args.patience,
    )
