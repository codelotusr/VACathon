# src/train.py
import argparse
import json
import pickle
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .model import GNNClassifier, _ece_binary


def load_graphs(path: str) -> List[Data]:
    """Load one or multiple shards of graphs into a single list."""
    # Allowlist PyG Data for PyTorch 2.6+ default weights_only=True
    torch.serialization.add_safe_globals([Data])

    p = Path(path)
    if p.exists():
        return torch.load(p, weights_only=False)

    # If base file doesn't exist, try shard pattern e.g. train.pt.000.pt, train.pt.001.pt, ...
    shards = sorted(p.parent.glob(p.stem + ".*.pt"))
    graphs: List[Data] = []
    for shard in shards:
        graphs.extend(torch.load(shard, weights_only=False))
    if not graphs:
        raise FileNotFoundError(f"No graphs found for {path} (or shards).")
    return graphs


def load_data(data_dir: str = "data/processed", batch_size: int = 32):
    """Load train/validation/test sets into PyG DataLoaders."""
    train_graphs = load_graphs(f"{data_dir}/train.pt")
    val_graphs = load_graphs(f"{data_dir}/validation.pt")
    test_graphs = load_graphs(f"{data_dir}/test.pt")

    return (
        DataLoader(train_graphs, batch_size=batch_size, shuffle=True),
        DataLoader(val_graphs, batch_size=batch_size),
        DataLoader(test_graphs, batch_size=batch_size),
    )


def _class_weights(loader) -> torch.Tensor:
    """Compute per-class weights from a DataLoader."""
    counts = np.zeros(2, dtype=float)
    for batch in loader:
        y = batch.y.numpy()
        counts[0] += (y == 0).sum()
        counts[1] += (y == 1).sum()
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    w = inv / inv.sum() * 2.0
    return torch.tensor(w, dtype=torch.float32)


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha: torch.Tensor | float = 0.5, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        if isinstance(self.alpha, torch.Tensor):
            at = self.alpha.to(logits.device)[targets]
        else:
            at = self.alpha
        loss = at * ((1 - pt) ** self.gamma) * ce
        return loss.mean()


def drop_edge(
    edge_index: torch.Tensor, edge_type: torch.Tensor | None = None, p: float = 0.1
):
    """Randomly drop edges for DropEdge regularization."""
    if p <= 0.0 or edge_index.numel() == 0:
        return edge_index, edge_type
    E = edge_index.size(1)
    keep = torch.rand(E, device=edge_index.device) > p
    ei2 = edge_index[:, keep]
    et2 = edge_type[keep] if edge_type is not None else None
    return ei2, et2


@torch.no_grad()
def probs_with_mc(
    model: GNNClassifier, loader, device_str: str, mc_passes: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (probs_pos, labels) with optional MC-Dropout averaging.
    """
    model.eval()

    def _enable_dropout(m: torch.nn.Module):
        if isinstance(m, torch.nn.Dropout):
            m.train()

    if mc_passes and mc_passes > 1:
        probs_sum: np.ndarray | None = None
        labels_all: np.ndarray | None = None
        for _ in range(mc_passes):
            model.eval()
            model.apply(_enable_dropout)
            all_probs, all_labels = [], []
            for batch in loader:
                batch = batch.to(device_str)
                logits = model(batch)
                p = F.softmax(logits, dim=1)[:, 1].detach().cpu().numpy()
                all_probs.append(p)
                all_labels.append(batch.y.cpu().numpy())
            p_pass = np.concatenate(all_probs)
            l_pass = np.concatenate(all_labels)
            if probs_sum is None:
                probs_sum = p_pass
                labels_all = l_pass
            else:
                probs_sum += p_pass
        assert probs_sum is not None and labels_all is not None
        return probs_sum / mc_passes, labels_all
    else:
        all_probs, all_labels = [], []
        for batch in loader:
            batch = batch.to(device_str)
            logits = model(batch)
            all_probs.append(F.softmax(logits, dim=1)[:, 1].cpu().numpy())
            all_labels.append(batch.y.cpu().numpy())
        return np.concatenate(all_probs), np.concatenate(all_labels)


def fit_isotonic(probs: np.ndarray, labels: np.ndarray) -> IsotonicRegression:
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(probs, labels.astype(float))
    return iso


def tune_threshold(
    probs: np.ndarray, labels: np.ndarray, target_precision: float = 0.85
):
    """
    Return (best_threshold, metrics_dict) choosing the smallest grid threshold
    achieving target precision, breaking ties by F1.
    """
    best_t, best_f1, best_metrics = 0.5, -1.0, {}
    for t in np.linspace(0.05, 0.95, 19):
        pred = (probs >= t).astype(int)
        pr, rc, f1, _ = precision_recall_fscore_support(
            labels, pred, average="binary", zero_division="warn"
        )
        if pr >= target_precision and f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
            best_metrics = {
                "precision": float(pr),
                "recall": float(rc),
                "f1": float(f1),
            }
    if best_f1 < 0:
        # fallback: maximize F1
        f1s = []
        ts = np.linspace(0.05, 0.95, 19)
        for t in ts:
            pred = (probs >= t).astype(int)
            pr, rc, f1, _ = precision_recall_fscore_support(
                labels, pred, average="binary", zero_division="warn"
            )
            f1s.append((f1, t, pr, rc))
        f1, t, pr, rc = max(f1s, key=lambda x: x[0])
        best_t = float(t)
        best_metrics = {"precision": float(pr), "recall": float(rc), "f1": float(f1)}
    coverage = float((probs >= best_t).mean())
    best_metrics["threshold"] = best_t
    best_metrics["coverage"] = coverage
    return best_t, best_metrics


def eval_full(
    model: GNNClassifier, loader, device_str: str, threshold: float = 0.5
) -> Tuple[float, float, float, float, float]:
    """Returns: acc, precision, recall, f1, ece (ECE computed at 0.5 by default)."""
    model.eval()
    all_probs, all_labels = [], []
    for batch in loader:
        batch = batch.to(device_str)
        logits = model(batch)
        probs = F.softmax(logits, dim=1)[:, 1].detach().cpu()
        all_probs.append(probs)
        all_labels.append(batch.y.cpu())
    probs_t = torch.cat(all_probs)
    labels_t = torch.cat(all_labels)
    pred = (probs_t >= threshold).int()
    acc = float((pred == labels_t).float().mean())
    pr, rc, f1, _ = precision_recall_fscore_support(
        labels_t.numpy(), pred.numpy(), average="binary", zero_division="warn"
    )
    ece = _ece_binary(probs_t, labels_t)
    return acc, float(pr), float(rc), float(f1), float(ece)


def train_model(
    epochs: int = 20,
    lr: float = 1e-3,
    hidden_dim: int = 128,
    batch_size: int = 32,
    data_dir: str = "data/processed",
    ckpt_path: str = "checkpoints/model.pt",
    patience: int = 10,
    use_focal: bool = True,
    dropedge_p: float = 0.1,
    target_precision: float = 0.85,
    mc_passes: int = 10,
    hardmine_start: int = 5,
    hardmine_factor: float = 2.0,
):
    # String device for pyright-friendly .to()
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    # Load vocab size for embedding
    vocab = json.load(open(f"{data_dir}/node_vocab.json"))
    model = GNNClassifier(len(vocab), hidden_dim=hidden_dim).to(device_str)

    train_loader, val_loader, test_loader = load_data(data_dir, batch_size)

    # Class weights + focal (helps imbalance)
    w = _class_weights(train_loader).to(device_str)
    if use_focal:
        criterion = FocalLoss(alpha=w, gamma=2.0)
    else:
        criterion = torch.nn.CrossEntropyLoss(weight=w)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3  # verbose not typed in some stubs
    )

    best_val_f1 = -1.0
    stale = 0
    best_plain_ckpt = ckpt_path

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0

        # (Optional) simple hard-example upsampling after a few epochs
        train_graphs = train_loader.dataset
        if (
            isinstance(train_graphs, list)
            and epoch >= hardmine_start
            and hardmine_factor > 1.0
        ):
            model.eval()
            mis_idx: List[int] = []
            with torch.no_grad():
                for idx, g in enumerate(train_graphs):
                    bg = Data(
                        x=g.x,
                        edge_index=g.edge_index,
                        edge_type=getattr(g, "edge_type", None),
                        y=g.y,
                        num_nodes=g.num_nodes,
                        node_feat=getattr(g, "node_feat", None),
                    )
                    bg.batch = torch.zeros(g.num_nodes, dtype=torch.long)
                    bg = bg.to(device_str)
                    p = F.softmax(model(bg), dim=1).argmax(dim=1).item()
                    if p != int(g.y.item()):
                        mis_idx.append(idx)

            aug = list(train_graphs)
            for idx in mis_idx:
                for _ in range(int(hardmine_factor) - 1):
                    aug.append(train_graphs[idx])
            train_loader = DataLoader(aug, batch_size=batch_size, shuffle=True)
            model.train()

        for batch in train_loader:
            batch = batch.to(device_str)

            # DropEdge regularization
            ei, et = drop_edge(
                batch.edge_index, getattr(batch, "edge_type", None), p=dropedge_p
            )
            orig_ei, orig_et = batch.edge_index, getattr(batch, "edge_type", None)
            batch.edge_index = ei
            if et is not None:
                batch.edge_type = et

            optimizer.zero_grad()
            logits = model(batch)
            loss = criterion(logits, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss.item())

            # restore
            batch.edge_index = orig_ei
            if orig_et is not None:
                batch.edge_type = orig_et

        # explicit average loss to avoid Optional issues
        denom = float(max(1, len(train_loader)))
        avg_loss = total_loss / denom

        val_acc, val_pr, val_rc, val_f1, val_ece = eval_full(
            model, val_loader, device_str, threshold=0.5
        )
        scheduler.step(val_f1)

        print(
            f"Epoch {epoch:03d} | Loss {avg_loss:.4f} | "
            f"Val Acc {val_acc:.4f} | P {val_pr:.4f} | R {val_rc:.4f} | F1 {val_f1:.4f} | ECE {val_ece:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            stale = 0
            Path(best_plain_ckpt).parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_plain_ckpt)
            print(
                f"  ✓ Saved new best (plain) → {best_plain_ckpt} (val_f1={best_val_f1:.4f})"
            )
        else:
            stale += 1
            if stale >= patience:
                print(
                    f"⏹️  Early stopping: no val F1 improvement for {patience} epochs."
                )
                break

    # Load best plain model
    model.load_state_dict(torch.load(best_plain_ckpt, map_location=device_str))

    # Temperature scaling
    model.set_temperature(val_loader, device_str)

    # Isotonic on validation probs
    val_probs_raw, val_labels = probs_with_mc(
        model, val_loader, device_str, mc_passes=mc_passes
    )
    iso = fit_isotonic(val_probs_raw, val_labels)
    cal_path = Path(best_plain_ckpt).with_suffix(".calib.pkl")
    with open(cal_path, "wb") as f:
        pickle.dump(iso, f)
    print(f"  ✓ Saved isotonic calibrator → {cal_path}")

    # Threshold tuning for target precision
    val_probs_cal = iso.predict(val_probs_raw)
    best_t, met = tune_threshold(
        val_probs_cal, val_labels, target_precision=target_precision
    )
    thr_path = Path(best_plain_ckpt).with_suffix(".threshold.txt")
    thr_path.write_text(str(best_t))
    print(
        f"  ✓ Tuned threshold t={best_t:.2f} (target P={target_precision:.2f}) "
        f"| P={met.get('precision'):.3f} R={met.get('recall'):.3f} F1={met.get('f1'):.3f} "
        f"| coverage={met.get('coverage'):.3f}"
    )

    # Final test (calibrated + tuned + MC-Dropout)
    test_probs_raw, test_labels = probs_with_mc(
        model, test_loader, device_str, mc_passes=mc_passes
    )
    test_probs_cal = iso.predict(test_probs_raw)
    test_pred = (test_probs_cal >= best_t).astype(int)

    acc = accuracy_score(test_labels, test_pred)
    pr, rc, f1, _ = precision_recall_fscore_support(
        test_labels, test_pred, average="binary", zero_division="warn"
    )

    # ECE on calibrated probs
    ece = _ece_binary(torch.tensor(test_probs_cal), torch.tensor(test_labels))

    print("\nFinal test (calibrated + tuned):")
    print(
        f"  Acc {acc:.4f} | P {pr:.4f} | R {rc:.4f} | F1 {f1:.4f} | ECE {ece:.4f} | thr {best_t:.2f}"
    )
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train supreme relational GNN on code vulnerability dataset"
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--data-dir", type=str, default="data/processed")
    parser.add_argument("--ckpt-path", type=str, default="checkpoints/rgcn_supreme.pt")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--no-focal", action="store_true")
    parser.add_argument("--dropedge-p", type=float, default=0.10)
    parser.add_argument("--target-precision", type=float, default=0.85)
    parser.add_argument("--mc-passes", type=int, default=10)
    parser.add_argument(
        "--hardmine-start",
        type=int,
        default=5,
        help="Start epoch for hard-example upsampling",
    )
    parser.add_argument(
        "--hardmine-factor",
        type=float,
        default=2.0,
        help="Duplication factor for hard examples",
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
        use_focal=not args.no_focal,
        dropedge_p=args.dropedge_p,
        target_precision=args.target_precision,
        mc_passes=args.mc_passes,
        hardmine_start=args.hardmine_start,
        hardmine_factor=args.hardmine_factor,
    )
