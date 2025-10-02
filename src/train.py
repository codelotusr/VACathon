import argparse
import json

import torch
from torch_geometric.loader import DataLoader

from .model import GNNClassifier


def load_data(data_dir="data/processed", batch_size=32):
    train_graphs = torch.load(f"{data_dir}/train.pt")
    val_graphs = torch.load(f"{data_dir}/validation.pt")
    test_graphs = torch.load(f"{data_dir}/test.pt")

    return (
        DataLoader(train_graphs, batch_size=batch_size, shuffle=True),
        DataLoader(val_graphs, batch_size=batch_size),
        DataLoader(test_graphs, batch_size=batch_size),
    )


def eval_model(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            pred = out.argmax(dim=1)
            correct += int((pred == batch.y).sum())
            total += batch.num_graphs
    return correct / total


def train_model(
    epochs=10,
    lr=1e-3,
    hidden_dim=64,
    batch_size=32,
    data_dir="data/processed",
    ckpt_path="model.pt",
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = json.load(open(f"{data_dir}/node_vocab.json"))
    model = GNNClassifier(len(vocab), hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = load_data(data_dir, batch_size)

    best_val_acc = 0.0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} | Loss {total_loss:.4f}")

        val_acc = eval_model(model, val_loader, device)
        print(f"Validation acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved new best model to {ckpt_path}")

    model.load_state_dict(torch.load(ckpt_path))
    test_acc = eval_model(model, test_loader, device)
    print(f"Final test acc (best model): {test_acc:.4f}")
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train GNN on code vulnerability dataset"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument(
        "--hidden-dim", type=int, default=64, help="Hidden dimension size"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/processed",
        help="Directory with .pt data files",
    )
    parser.add_argument(
        "--ckpt-path", type=str, default="model.pt", help="Path to save best checkpoint"
    )

    args = parser.parse_args()

    train_model(
        epochs=args.epochs,
        lr=args.lr,
        hidden_dim=args.hidden_dim,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        ckpt_path=args.ckpt_path,
    )
