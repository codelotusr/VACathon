# src/data.py
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import torch
from datasets import Dataset, IterableDataset, load_dataset
from torch_geometric.data import Data

from .ast_utils import (
    ast_nodes_and_edges,
    ast_nodes_and_edges_with_types,
    make_parser,
    normalize_code,
)

# -------------------
# Defaults / config
# -------------------
DEFAULT_HF_ID = "bstee615/bigvul"
VALID_SPLITS = ("train", "validation", "test")

# Fallback heuristics (only used if func_before/func_after missing)
LANG_FIELDS: Tuple[str, ...] = ("lang", "language")
PATH_FIELDS: Tuple[str, ...] = ("path", "filepath", "file", "filename")


# -------------------
# Small helpers
# -------------------
def _infer_lang_from_strings(value: str | None) -> str:
    if not isinstance(value, str):
        return "c"
    lv = value.lower()
    if "cpp" in lv or "c++" in lv:
        return "cpp"
    if lv.endswith((".cpp", ".cc", ".cxx", ".hpp", ".hh")):
        return "cpp"
    return "c"


def _infer_lang(row: Dict[str, Any]) -> str:
    # Prefer explicit 'lang'
    lang = row.get("lang")
    if isinstance(lang, str):
        return _infer_lang_from_strings(lang)
    # Fall back to file path style fields
    for k in PATH_FIELDS:
        if isinstance(row.get(k), str):
            return _infer_lang_from_strings(row[k])
    return "c"


def _codes_from_row(row: Dict[str, Any]) -> list[tuple[str, int, str]]:
    """
    Big-Vul row -> [(code, label, lang), ...]
    - func_before -> label 1 (vulnerable)
    - func_after  -> label 0 (safe)
    If 'lang' present, use it; else infer from path; default 'c'.
    """
    out: list[tuple[str, int, str]] = []
    lang = _infer_lang(row)

    fb = row.get("func_before")
    if isinstance(fb, str) and fb.strip():
        out.append((fb, 1, lang))

    fa = row.get("func_after")
    if isinstance(fa, str) and fa.strip():
        out.append((fa, 0, lang))

    return out


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
) -> Tuple[List[str], List[Tuple[int, int]], List[int]]:
    toks = [t for t in code.replace("\n", " ").split(" ") if t]
    if not toks:
        return ["<EMPTY_FUNC>"], [], []
    toks = toks[:max_tokens]
    edges = [(i, i + 1) for i in range(len(toks) - 1)]
    etypes = [1] * len(edges)  # sequential only
    return toks, edges, etypes


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
# Vocab (single pass over train)
# -------------------
def build_vocab_streaming(
    hf_id: str,
    max_nodes: int,
    min_freq: int,
    streaming: bool,
    limit: int | None,
    add_seq_edges: bool,
) -> Dict[str, int]:
    c_parser = make_parser("c")
    cpp_parser = make_parser("cpp")
    counter: Counter[str] = Counter()
    stats = Counter()

    for row in iter_split(hf_id, "train", limit, streaming):
        codes = _codes_from_row(row)
        if not codes:
            stats["no_code"] += 1
            continue

        for code, _label, lang in codes:
            parser = c_parser if lang == "c" else cpp_parser
            # For vocab speed, the simpler extractor is fine
            node_types, edges = ast_nodes_and_edges(code, parser, max_nodes=max_nodes)
            if not node_types:
                other = cpp_parser if parser is c_parser else c_parser
                node_types, edges = ast_nodes_and_edges(
                    code, other, max_nodes=max_nodes
                )
            if not node_types:
                node_types, edges, _et = _fallback_graph(code)
                if not node_types:
                    stats["fallback_fail"] += 1
                    continue

            if add_seq_edges and node_types:
                edges = edges + [(i, i + 1) for i in range(len(node_types) - 1)]

            counter.update(node_types)
            stats["ok"] += 1

    print(f"[train] vocab pass stats: {dict(stats)}")

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
    add_seq_edges: bool,
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
    stats = Counter()

    def _flush():
        nonlocal graphs, shard_idx, paths, n_written
        if not graphs:
            return
        if shard_idx == 0:
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
        graphs.clear()
        shard_idx += 1

    for i, row in enumerate(iter_split(hf_id, split, limit, streaming)):
        codes = _codes_from_row(row)
        if not codes:
            stats["no_code"] += 1
            continue

        for code, label, lang in codes:
            parser = c_parser if lang == "c" else cpp_parser

            # --- NEW: light normalization helps reduce noisy parse tokens ---
            code_n = normalize_code(code)

            # Build multi-relation graph (AST/seq/sibling/ident)
            node_types, edges, edge_types = ast_nodes_and_edges_with_types(
                code_n, parser, max_nodes=max_nodes
            )
            if not node_types:
                # try other language parser
                other = cpp_parser if parser is c_parser else c_parser
                node_types, edges, edge_types = ast_nodes_and_edges_with_types(
                    code_n, other, max_nodes=max_nodes
                )

            if not node_types:
                # final fallback: token chain
                node_types, edges, edge_types = _fallback_graph(code_n)
                if not node_types:
                    stats["fallback_fail"] += 1
                    continue

            if add_seq_edges and node_types:
                # already included in _with_types; keep for backward-compat if flag is used
                pass

            x = torch.tensor([vocab.get(t, 0) for t in node_types], dtype=torch.long)
            edge_index = edges_to_edge_index(edges, len(node_types))

            # Align edge_types with filtered edge_index
            if len(edges) == edge_index.shape[1]:
                et = torch.tensor(edge_types, dtype=torch.long)
            else:
                # recompute kept types for valid edges
                keep: List[int] = []
                k = 0
                for a, b in edges:
                    if 0 <= a < len(node_types) and 0 <= b < len(node_types):
                        keep.append(edge_types[k])
                    k += 1
                et = torch.tensor(keep, dtype=torch.long)
                if et.numel() != edge_index.shape[1]:
                    et = et[: edge_index.shape[1]]

            y = torch.tensor([label], dtype=torch.long)
            graphs.append(
                Data(
                    x=x,
                    edge_index=edge_index,
                    edge_type=et,
                    y=y,
                    num_nodes=len(node_types),
                )
            )
            stats["ok"] += 1

            if len(graphs) >= shard_size:
                _flush()

    _flush()
    print(f"[{split}] stats: {dict(stats)}")
    return n_written, paths


# -------------------
# CLI
# -------------------
def main():
    ap = argparse.ArgumentParser(
        description="Build PyG .pt graphs from HuggingFace Big-Vul (func_before/func_after)."
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

    add_seq_edges = not args.no_seq_edges

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
        add_seq_edges=add_seq_edges,
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
            add_seq_edges=add_seq_edges,
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
