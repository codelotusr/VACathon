# src/visualize_graph.py
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# --- allowlist PyG classes for torch.load(weights_only=True default in PyTorch 2.6+) ---
from torch.serialization import add_safe_globals

try:
    from torch_geometric.data import Data, DataEdgeAttr, HeteroData  # PyG >= 2.5
except Exception:
    # Older PyG may not have DataEdgeAttr/HeteroData
    Data = None
    DataEdgeAttr = None
    HeteroData = None

_allowed = [cls for cls in (Data, DataEdgeAttr, HeteroData) if cls is not None]
if _allowed:
    add_safe_globals(_allowed)

try:
    from graphviz import Digraph

    GV_AVAILABLE = True
except Exception:
    GV_AVAILABLE = False


def load_vocab(vocab_path: Path) -> dict[int, str]:
    vocab = json.loads(vocab_path.read_text())
    # invert: id -> type
    inv = {int(v): k for k, v in vocab.items()}
    return inv


def pick_edges_parent_child_only(edge_index, num_nodes: int) -> list[tuple[int, int]]:
    """
    You saved edges as bidirectional (parent->child and child->parent).
    For a cleaner tree view, keep only one direction by deduping pairs.
    We don't know actual parent/child direction here, so just keep (u,v) where u<v.
    It's purely for visualization (structure stays readable).
    """
    ei = edge_index.cpu().numpy()
    edges = []
    seen = set()
    for u, v in zip(ei[0], ei[1]):
        if 0 <= u < num_nodes and 0 <= v < num_nodes:
            a, b = int(u), int(v)
            key = (min(a, b), max(a, b))
            if key not in seen and a != b:
                seen.add(key)
                edges.append(key)
    return edges


def save_dot(
    node_labels: list[str],
    edges: list[tuple[int, int]],
    out_dot: Path,
    title: str = "VACron AST",
    max_nodes: int | None = 120,
):
    n = len(node_labels)
    keep_n = n if (max_nodes is None or max_nodes >= n) else max_nodes
    label_map = {i: node_labels[i] for i in range(keep_n)}

    edges = [(u, v) for (u, v) in edges if u < keep_n and v < keep_n]

    lines = [
        f"digraph G {{",
        f'  label="{title}"; labelloc=top; rankdir=TB; node [shape=box, fontsize=10];',
    ]
    for i in range(keep_n):
        safe = label_map[i].replace('"', '\\"')
        lines.append(f'  {i} [label="{i}: {safe}"];')
    for u, v in edges:
        lines.append(f"  {u} -> {v};")
    lines.append("}")
    out_dot.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--pt",
        required=True,
        help="Path to .pt file (train.pt / validation.pt / test.pt)",
    )
    ap.add_argument(
        "--vocab",
        default="data/processed/node_vocab.json",
        help="Path to node_vocab.json",
    )
    ap.add_argument(
        "--index", type=int, default=0, help="Graph index inside the .pt list"
    )
    ap.add_argument(
        "--max_nodes",
        type=int,
        default=120,
        help="Limit nodes in the rendering for readability",
    )
    ap.add_argument(
        "--out",
        default="graph",
        help="Output basename (writes .dot and optionally .png)",
    )
    args = ap.parse_args()

    pt_path = Path(args.pt)
    vocab_path = Path(args.vocab)
    out_base = Path(args.out)

    try:
        graphs = torch.load(pt_path)
    except Exception:
        graphs = torch.load(pt_path, weights_only=False)

    if not graphs:
        raise SystemExit("No graphs found in the .pt file.")

    if args.index < 0 or args.index >= len(graphs):
        raise SystemExit(f"--index out of range 0..{len(graphs)-1}")

    g = graphs[args.index]
    if not hasattr(g, "x") or not hasattr(g, "edge_index"):
        raise SystemExit("Selected item is not a PyG Data graph with x/edge_index.")

    inv_vocab = load_vocab(vocab_path)
    node_labels = [inv_vocab.get(int(t.item()), "<UNK>") for t in g.x]
    edges = pick_edges_parent_child_only(g.edge_index, g.num_nodes)

    out_base.parent.mkdir(parents=True, exist_ok=True)

    dot_path = out_base.with_suffix(".dot")
    save_dot(
        node_labels,
        edges,
        dot_path,
        title=f"VACron AST (idx={args.index})",
        max_nodes=args.max_nodes,
    )
    print(f"Wrote DOT: {dot_path}")

    if GV_AVAILABLE:
        dot = Digraph(comment="VACron AST")
        dot.attr(rankdir="TB")
        dot.attr(label=f"VACron AST (idx={args.index})", labelloc="top")
        keep_n = (
            len(node_labels)
            if args.max_nodes is None
            else min(len(node_labels), args.max_nodes)
        )

        for i in range(keep_n):
            dot.node(str(i), f"{i}: {node_labels[i]}", shape="box")

        for u, v in edges:
            if u < keep_n and v < keep_n:
                dot.edge(str(u), str(v))

        # Renders to PNG next to the .dot
        png_path = out_base.with_suffix(".png")
        dot.render(
            filename=out_base.name,
            format="png",
            directory=str(out_base.parent),
            cleanup=True,
        )
        print(f"Wrote PNG: {png_path}")
    else:
        print("graphviz package or system binary not found.")


if __name__ == "__main__":
    main()
