# src/data.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import pandas as pd
import torch
from torch_geometric.data import Data

from .ast_utils import ast_nodes_and_edges, make_c_parser

VALID_SPLITS = {"train", "validation", "test"}

BIGVUL_SPLITS = {
    "train": "hf://datasets/bstee615/bigvul/data/train-00000-of-00001-c6410a8bb202ca06.parquet",
    "validation": "hf://datasets/bstee615/bigvul/data/validation-00000-of-00001-d21ad392180d1f79.parquet",
    "test": "hf://datasets/bstee615/bigvul/data/test-00000-of-00001-d20b0e7149fa6eeb.parquet",
}

CODE_FIELDS: Tuple[str, ...] = ("func", "code", "function", "functionSource")
LABEL_FIELDS: Tuple[str, ...] = ("target", "label", "vul")


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


def load_bigvul_frame(split: str, limit: int | None = None) -> Iterable[Any]:
    if split not in VALID_SPLITS:
        raise ValueError(f"Split must be one of {sorted(VALID_SPLITS)}; got {split!r}")

    path = BIGVUL_SPLITS.get(split)
    if path is None:
        raise ValueError(f"No configured path for split {split!r}")

    frame = pd.read_parquet(path)
    if limit is not None:
        frame = frame.head(limit)

    return frame.itertuples(index=False)


def parse_split(
    split: str, limit: int | None = None
) -> List[Tuple[List[str], List[Tuple[int, int]], int]]:
    rows = load_bigvul_frame(split, limit)

    parser = make_c_parser()
    out: List[Tuple[List[str], List[Tuple[int, int]], int]] = []

    for row in rows:
        code = _extract_first_attr(row, CODE_FIELDS)
        if not isinstance(code, str) or not code.strip():
            continue

        label = _extract_label(row)

        node_types, edges = ast_nodes_and_edges(code, parser)
        if node_types:
            out.append((node_types, edges, label))

    return out


def _extract_first_attr(row: Any, candidates: Tuple[str, ...]) -> Any:
    for attr in candidates:
        if hasattr(row, attr):
            value = getattr(row, attr)
            if value is not None:
                return value
    return None


def _extract_label(row: Any) -> int:
    for attr in LABEL_FIELDS:
        if hasattr(row, attr):
            value = getattr(row, attr)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    continue
    # Default to benign
    return 0


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
