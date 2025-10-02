# src/data.py
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple, Union

import torch
from datasets import Dataset, IterableDataset, load_dataset
from torch_geometric.data import Data

from .ast_utils import ast_nodes_and_edges, make_parser

# -------------------
# Defaults / config
# -------------------
DEFAULT_HF_ID = "bstee615/bigvul"
VALID_SPLITS = ("train", "validation", "test")

# Candidate columns used across variants of Big-Vul or mirrors
CODE_FIELDS: Tuple[str, ...] = ("func", "code", "function", "functionSource")
LABEL_FIELDS: Tuple[str, ...] = ("target", "label", "vul")
LANG_FIELDS: Tuple[str, ...] = ("lang", "language")
PATH_FIELDS: Tuple[str, ...] = ("path", "filepath", "file", "filename")


# -------------------
# Small helpers
# -------------------
def _first_present(row: Dict[str, Any], names: Tuple[str, ...]) -> Any:
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return None


def _extract_label(row: Dict[str, Any]) -> int:
    v = _first_present(row, LABEL_FIELDS)
    try:
        return int(v) if v is not None else 0
    except Exception:
        return 0


def _infer_lang(row: Dict[str, Any]) -> str:
    # explicit column
    v = _first_present(row, LANG_FIELDS)
    if isinstance(v, str):
        lv = v.lower()
        if "cpp" in lv or "c++" in lv:
            return "cpp"
        if lv == "c":
            return "c"
    # infer from path
    v = _first_present(row, PATH_FIELDS)
    if isinstance(v, str):
        lv = v.lower()
        if lv.endswith((".cpp", ".cc", ".cxx", ".hpp", ".hh")):
            return "cpp"
        if lv.endswith((".c", ".h")):
            return "c"
    return "c"


def edges_to_edge_index(edges: List[Tuple[int, int]], n: int) -> torch.Tensor:
    if not edges:
        return torch.zeros((2, 0), dtype=torch.long)
    src = [a for a, b in edges] + [b for a, b in edges]
    dst = [b for a, b in edges] + [a for a, b in edges]
    ei = torch.tensor([src, dst], dtype=torch.long)
    m = (ei[0] >= 0) & (ei[1] >= 0) & (ei[0] < n) & (ei[1] < n)
    return ei[:, m]


def _fallback_graph(
    code: str, max_tokens: int = 512
) -> Tuple[List[str], List[Tuple[int, int]]]:
    toks = [t for t in code.replace("\n", " ").split(" ") if t]
    if not toks:
        return ["<EMPTY_FUNC>"], []
    toks = toks[:max_tokens]
    edges = [(i, i + 1) for i in range(len(toks) - 1)]
    return toks, edges


# -------------------
# HuggingFace iteration
# -------------------
def iter_split(
    hf_id: str,
    split: str,
    limit: int | None,
    streaming: bool,
) -> Iterator[Dict[str, Any]]:
    """
    Yields dict rows from HF datasets in either in-memory (Dataset) or streaming (IterableDataset) mode.
    """
    if streaming:
        ds: IterableDataset = load_dataset(hf_id, split=split, streaming=True)  # type: ignore[assignment]
        it = iter(ds)
        if limit is None:
            yield from it
        else:
            for i, ex in enumerate(it):
                if i >= limit:
                    break
                yield ex
    else:
        d: Dataset = load_dataset(hf_id, split=split)  # type: ignore[assignment]
        if limit is not None:
            d = d.select(range(min(limit, len(d))))
        for ex in d:
            yield ex


# -------------------
# Vocab (streaming over train once)
# -------------------
def build_vocab_streaming(
    hf_id: str,
    max_nodes: int,
    min_freq: int,
    streaming: bool,
    limit: int | None,
    seq_edges: bool,
) -> Dict[str, int]:
    """
    Single pass over TRAIN split; parse ASTs and count node types.
    Memory efficient (keeps just a Counter).
    """
    c_parser = make_parser("c")
    cpp_parser = make_parser("cpp")
    counter: Counter[str] = Counter()

    for row in iter_split(hf_id, "train", limit, streaming):
        code = _first_present(row, CODE_FIELDS)
        if not isinstance(code, str) or code.strip() == "":
            continue
        lang = _infer_lang(row)
        parser = c_parser if lang == "c" else cpp_parser

        node_types, edges = ast_nodes_and_edges(code, parser, max_nodes=max_nodes)
        if not node_types:
            # try other language parser before fallback
            other = cpp_parser if parser is c_parser else c_parser
            node_types, edges = ast_nodes_and_edges(code, other, max_nodes=max_nodes)
        if not node_types:
            node_types, edges = _fallback_graph(code)

        # optional sequential edges (doesn't affect vocab, but we keep the same logic between passes)
        if seq_edges and node_types:
            edges = edges + [(i, i + 1) for i in range(len(node_types) - 1)]

        counter.update(node_types)

    vocab: Dict[str, int] = {"<UNK>": 0}
    for t, c in counter.items():
        if c >= min_freq and t not in vocab:
            vocab[t] = len(vocab)
    return vocab


# -------------------
# Encode + write (sharded, streaming)
# -------------------
def write_split(
    hf_id: str,
    split: str,
    out_base: Path,
    vocab: Dict[str, int],
    max_nodes: int,
    streaming: bool,
    limit: int | None,
    shard_size: int,
    seq_edges: bool,
) -> Tuple[int, List[Path]]:
    """
    Iterates a split, parses & encodes graphs, and writes shards:
      out_base = data/processed/train.pt
      -> train.pt (if < shard_size) or train.pt.000.pt, train.pt.001.pt, ...
    Returns (num_graphs, shard_paths).
    """
    c_parser = make_parser("c")
    cpp_parser = make_parser("cpp")
    paths: List[Path] = []
    graphs: List[Data] = []
    shard_idx = 0
    n_written = 0

    def _flush():
        nonlocal graphs, shard_idx, paths, n_written
        if not graphs:
            return
        if shard_idx == 0:
            # if only one shard at the end, we'll overwrite to base; here we tentatively shard
            shard_path = (
                out_base
                if len(graphs) < shard_size and n_written == 0
                else out_base.with_suffix(f".{shard_idx:03d}.pt")
            )
        else:
            shard_path = out_base.with_suffix(f".{shard_idx:03d}.pt")
        torch.save(graphs, shard_path)
        paths.append(shard_path)
        n_written += len(graphs)
        graphs = []
        shard_idx += 1

    for i, row in enumerate(iter_split(hf_id, split, limit, streaming)):
        code = _first_present(row, CODE_FIELDS)
        if not isinstance(code, str) or not code.strip():
            continue
        y = torch.tensor([_extract_label(row)], dtype=torch.long)

        lang = _infer_lang(row)
        parser = c_parser if lang == "c" else cpp_parser

        node_types, edges = ast_nodes_and_edges(code, parser, max_nodes=max_nodes)
        if not node_types:
            other = cpp_parser if parser is c_parser else c_parser
            node_types, edges = ast_nodes_and_edges(code, other, max_nodes=max_nodes)
        if not node_types:
            node_types, edges = _fallback_graph(code)

        if seq_edges and node_types:
            edges = edges + [(j, j + 1) for j in range(len(node_types) - 1)]

        x = torch.tensor([vocab.get(t, 0) for t in node_types], dtype=torch.long)
        edge_index = edges_to_edge_index(edges, len(node_types))
        g = Data(x=x, edge_index=edge_index, y=y, num_nodes=len(node_types))
        graphs.append(g)

        if len(graphs) >= shard_size:
            _flush()

    _flush()
    return n_written, paths


# -------------------
# CLI
# -------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build PyG .pt graphs from a HuggingFace dataset (Big-Vul)."
    )
    ap.add_argument(
        "--hf-id",
        type=str,
        default=DEFAULT_HF_ID,
        help="HuggingFace dataset id (e.g., bstee615/bigvul)",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        default="data/processed",
        help="Where to write .pt shards + vocab",
    )
    ap.add_argument(
        "--streaming",
        action="store_true",
        help="Use HF streaming (low memory, recommended for full dataset)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap examples per split (for dev). No cap if omitted.",
    )
    ap.add_argument(
        "--max-nodes",
        type=int,
        default=4000,
        help="Max AST nodes per function (truncate beyond)",
    )
    ap.add_argument(
        "--min-freq",
        type=int,
        default=1,
        help="Min frequency to include a node type in vocab",
    )
    ap.add_argument(
        "--shard-size",
        type=int,
        default=50000,
        help="Graphs per .pt shard (controls file size)",
    )
    ap.add_argument(
        "--no-seq-edges",
        action="store_true",
        help="Disable sequential edges between preorder-adjacent nodes",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    seq_edges = not args.no_seq_edges

    # 1) Build vocab from TRAIN only (single streaming pass)
    print(
        f"[1/3] Building vocab from {args.hf_id}::train (streaming={args.streaming}, limit={args.limit}) ..."
    )
    vocab = build_vocab_streaming(
        hf_id=args.hf_id,
        max_nodes=args.max_nodes,
        min_freq=args.min_freq,
        streaming=args.streaming,
        limit=args.limit,
        seq_edges=seq_edges,
    )
    (out_dir / "node_vocab.json").write_text(json.dumps(vocab, indent=2))
    print(f"    Vocab size = {len(vocab)}  (saved to {out_dir/'node_vocab.json'})")

    # 2) Encode & write train/validation/test (streaming, sharded)
    total_counts: Dict[str, int] = {}
    split_to_base = {
        "train": out_dir / "train.pt",
        "validation": out_dir / "validation.pt",
        "test": out_dir / "test.pt",
    }

    for split in VALID_SPLITS:
        print(
            f"[2/3] Processing split: {split} (streaming={args.streaming}, limit={args.limit}) ..."
        )
        n, paths = write_split(
            hf_id=args.hf_id,
            split=split,
            out_base=split_to_base[split],
            vocab=vocab,
            max_nodes=args.max_nodes,
            streaming=args.streaming,
            limit=args.limit,
            shard_size=args.shard_size,
            seq_edges=seq_edges,
        )
        total_counts[split] = n
        if (
            len(paths) == 1
            and paths[0].suffix == ".pt"
            and paths[0].name.endswith(".pt")
        ):
            print(f"    Wrote {n} graphs → {paths[0]}")
        else:
            print(f"    Wrote {n} graphs across {len(paths)} shards:")
            for p in paths:
                print(f"      - {p}")

    # 3) Summary
    print("[3/3] Done.")
    print("Counts:", total_counts)
    print(
        f"Files in {out_dir}: node_vocab.json + {{train,validation,test}}.pt (or sharded .pt.*.pt)"
    )
    print("Tip: the trainer already loads sharded files automatically (glob pattern).")


if __name__ == "__main__":
    main()
