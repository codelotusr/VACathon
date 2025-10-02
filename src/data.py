# src/data.py
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch
from datasets import Dataset, IterableDataset, load_dataset
from torch_geometric.data import Data

from .ast_utils import ast_nodes_and_edges  # back-compat wrapper in ast_utils.py
from .ast_utils import (
    ast_nodes_and_edges_with_types,
    detect_sink_lines,
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
) -> Generator[Any, None, None]:
    """
    Yields dataset rows. We keep the yield type as Any to be robust across HF backends.
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
            # Keep vocab pass fast; wrapper returns (labels, edges)
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
# Feature extraction
# -------------------
def _compute_node_features(
    node_types: List[str],
    edges: List[Tuple[int, int]],
    parent_of: Dict[int, Optional[int]],
) -> torch.Tensor:
    n = len(node_types)
    deg_in = [0] * n
    deg_out = [0] * n
    for a, b in edges:
        if 0 <= a < n and 0 <= b < n:
            deg_out[a] += 1
            deg_in[b] += 1

    # condition context flag via parent chain
    def in_cond(idx: int) -> int:
        from .ast_utils import _parent_chain_contains_cond

        return _parent_chain_contains_cond(idx, parent_of, node_types)

    feats = []
    for i, t in enumerate(node_types):
        is_id = int(t == "identifier")
        is_num = int("number" in t or t == "number_literal")
        is_str = int("string" in t or t == "string_literal")
        feats.append(
            [
                is_id,
                is_num,
                is_str,
                in_cond(i),
                deg_in[i],
                deg_out[i],
            ]
        )
    return torch.tensor(feats, dtype=torch.float)


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
    slice_k: int,
    slice_only_if_sink: bool,
) -> Tuple[int, List[Path]]:
    """
    Iterates a split, parses & encodes graphs, optionally slices around sinks, and writes shards:
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
            code_n = normalize_code(code)

            # Build multi-relation graph
            node_types, edges, edge_types, parent_of = ast_nodes_and_edges_with_types(
                code_n, parser, max_nodes=max_nodes
            )
            if not node_types:
                other = cpp_parser if parser is c_parser else c_parser
                node_types, edges, edge_types, parent_of = (
                    ast_nodes_and_edges_with_types(code_n, other, max_nodes=max_nodes)
                )

            # Fallback to token chain
            if not node_types:
                node_types, edges, edge_types = _fallback_graph(code_n)
                # Provide a typed parent map: set parent=None for all nodes (safe default)
                parent_of = {idx: None for idx in range(len(node_types))}
                if not node_types:
                    stats["fallback_fail"] += 1
                    continue

            n = len(node_types)

            # --- Optional slicing around sinks ---
            # Use bool-typed mask; if slicing applies, rebuild parent_of accordingly.
            mask: List[bool] = [True] * n
            if slice_k >= 0:
                sink_lines = detect_sink_lines(code_n)
                if sink_lines or not slice_only_if_sink:
                    if sink_lines:
                        from collections import deque

                        adj: List[List[int]] = [[] for _ in range(n)]
                        for a, b in edges:
                            if 0 <= a < n and 0 <= b < n:
                                adj[a].append(b)
                                adj[b].append(a)
                        dist = [-1] * n
                        q = deque()
                        # coarse seeds = all nodes (you may restrict to identifiers)
                        for s in range(n):
                            dist[s] = 0
                            q.append(s)
                        while q:
                            u = q.popleft()
                            if dist[u] >= slice_k:
                                continue
                            for v in adj[u]:
                                if dist[v] == -1:
                                    dist[v] = dist[u] + 1
                                    q.append(v)
                        mask = [d != -1 and d <= slice_k for d in dist]

            # Apply mask if it meaningfully reduces the graph
            if any(not m for m in mask) and any(mask):
                idx_map: Dict[int, int] = {}
                new_types: List[str] = []
                new_idx = 0
                for i_old, keep in enumerate(mask):
                    if keep:
                        idx_map[i_old] = new_idx
                        new_types.append(node_types[i_old])
                        new_idx += 1

                new_edges: List[Tuple[int, int]] = []
                new_et: List[int] = []
                for (a, b), et in zip(edges, edge_types):
                    if a in idx_map and b in idx_map:
                        new_edges.append((idx_map[a], idx_map[b]))
                        new_et.append(et)

                # rebuild parent_of for kept nodes
                parent_of_new: Dict[int, Optional[int]] = {}
                for old_child, old_parent in parent_of.items():
                    if old_child in idx_map:
                        child_new = idx_map[old_child]
                        if old_parent is None:
                            parent_of_new[child_new] = None
                        else:
                            parent_of_new[child_new] = idx_map.get(old_parent, None)

                node_types, edges, edge_types = new_types, new_edges, new_et
                parent_of = parent_of_new
                n = len(node_types)

            # Node features
            feat = _compute_node_features(node_types, edges, parent_of)

            if add_seq_edges and node_types:
                # already included in _with_types; keep for BC if flag present
                pass

            x = torch.tensor([vocab.get(t, 0) for t in node_types], dtype=torch.long)
            edge_index = edges_to_edge_index(edges, n)

            # Align edge_types with filtered edge_index
            if len(edges) == edge_index.shape[1]:
                et = torch.tensor(edge_types, dtype=torch.long)
            else:
                keep: List[int] = []
                k = 0
                for a, b in edges:
                    if 0 <= a < n and 0 <= b < n:
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
                    num_nodes=n,
                    node_feat=feat,
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
    ap.add_argument("--hf-id", type=str, default=DEFAULT_HF_ID)
    ap.add_argument("--out_dir", type=str, default="data/processed")
    ap.add_argument("--streaming", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-nodes", type=int, default=4000)
    ap.add_argument("--min-freq", type=int, default=1)
    ap.add_argument("--shard-size", type=int, default=50000)
    ap.add_argument("--no-seq-edges", action="store_true")
    ap.add_argument(
        "--slice-k", type=int, default=-1, help="k-hop slice around sinks (-1 disables)"
    )
    ap.add_argument(
        "--slice-only-if-sink",
        action="store_true",
        help="Only slice when a sink is found",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    add_seq_edges = not args.no_seq_edges

    # 1) Build vocab from TRAIN only
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

    # 2) Encode & write splits
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
            slice_k=args.slice_k,
            slice_only_if_sink=args.slice_only_if_sink,
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
    print("Tip: the trainer loads sharded files automatically.")


if __name__ == "__main__":
    main()
