# src/data.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, cast

import torch
from datasets import Dataset, DatasetDict, load_dataset
from torch_geometric.data import Data

from .ast_utils import ast_nodes_and_edges, make_c_parser

VALID_SPLITS = {"train", "validation", "test"}


def build_vocab(node_type_lists: List[List[str]], min_freq: int = 1) -> Dict[str, int]:
    from collections import Counter

    counter = Counter(t for lst in node_type_lists for t in lst)
    vocab = {"<UNK>": 0}
    for t, c in counter.items():
        if c >= min_freq and t not in vocab:
            vocab[t] = len(vocab)
    return vocab


def encode_nodes(node_types: List[str], vocab: Dict[str, int]) -> torch.Tensor:
    return torch.tensor([vocab.get(t, 0) for t in node_types], dtype=torch.long)


def edges_to_edge_index(edges: List[Tuple[int, int]], num_nodes: int) -> torch.Tensor:
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    src = [a for a, b in edges] + [b for a, b in edges]
    dst = [b for a, b in edges] + [a for a, b in edges]
    ei = torch.tensor([src, dst], dtype=torch.long)
    mask = (ei[0] >= 0) & (ei[1] >= 0) & (ei[0] < num_nodes) & (ei[1] < num_nodes)
    return ei[:, mask]


def parse_split(
    split: str, limit: int | None = None
) -> List[Tuple[List[str], List[Tuple[int, int]], int]]:
    if split not in VALID_SPLITS:
        raise ValueError(f"Split must be one of {sorted(VALID_SPLITS)}; got {split!r}")

    ds_any = load_dataset("google/code_x_glue_cc_defect_detection", split=split)

    if isinstance(ds_any, DatasetDict):
        raise RuntimeError(
            f"Got DatasetDict for split={split!r}. Available keys: {list(ds_any.keys())}. "
            "Use a valid split name: 'train', 'validation', or 'test'."
        )

    ds: Dataset = cast(Dataset, ds_any)

    if limit is not None:
        ds = ds.select(range(min(limit, len(ds))))

    parser = make_c_parser()
    out: List[Tuple[List[str], List[Tuple[int, int]], int]] = []

    for i in range(len(ds)):
        row = cast(Dict[str, Any], ds[i])
        code = cast(str, row.get("func") or row.get("code") or "")
        label = int(row.get("target", 0))

        node_types, edges = ast_nodes_and_edges(code, parser)
        if node_types:
            out.append((node_types, edges, label))

    return out


def write_split(records, vocab: dict, out_path: Path):
    graphs: List[Data] = []
    for node_types, edges, label in records:
        x = encode_nodes(node_types, vocab)
        edge_index = edges_to_edge_index(edges, len(node_types))
        y = torch.tensor([label], dtype=torch.long)
        graphs.append(Data(x=x, edge_index=edge_index, y=y, num_nodes=len(node_types)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(graphs, out_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/processed")
    ap.add_argument("--limit", type=int, default=None, help="cap per split (dev speed)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train: parse + vocab
    train_recs = parse_split("train", limit=args.limit)
    vocab = build_vocab([nts for nts, _, _ in train_recs])
    (out_dir / "node_vocab.json").write_text(json.dumps(vocab, indent=2))

    # Save train
    write_split(train_recs, vocab, out_dir / "train.pt")

    # Validation / Test with same vocab
    val_recs = parse_split("validation", limit=args.limit)
    test_recs = parse_split("test", limit=args.limit)
    write_split(val_recs, vocab, out_dir / "validation.pt")
    write_split(test_recs, vocab, out_dir / "test.pt")

    print(f"Saved: {out_dir}/train.pt, validation.pt, test.pt + node_vocab.json")


if __name__ == "__main__":
    main()
