# src/loaders.py
import json, torch
from torch_geometric.loader import DataLoader

# paths from your script
train_graphs = torch.load("data/processed/train.pt")      # list[Data]
val_graphs   = torch.load("data/processed/validation.pt") # list[Data]
test_graphs  = torch.load("data/processed/test.pt")       # list[Data]
vocab_size   = len(json.load(open("data/processed/node_vocab.json")))

train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_graphs,   batch_size=64, shuffle=False)
test_loader  = DataLoader(test_graphs,  batch_size=64, shuffle=False)
