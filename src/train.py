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


def train_model(epochs=10, lr=1e-3, hidden_dim=64):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = json.load(open("data/processed/node_vocab.json"))
    model = GNNClassifier(len(vocab), hidden_dim=hidden_dim).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    train_loader, val_loader, test_loader = load_data()

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

        acc = eval_model(model, val_loader, device)
        print(f"Validation acc: {acc:.4f}")

    print("Final test acc:", eval_model(model, test_loader, device))
    return model


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


if __name__ == "__main__":
    train_model(epochs=100, lr=1e-3)
