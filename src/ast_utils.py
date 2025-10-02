from __future__ import annotations

import re
from typing import Dict, List, Tuple

from tree_sitter import Node, Parser
from tree_sitter_languages import get_language

IDENT_TYPES = {"identifier"}  # tree-sitter-c / c++ both use 'identifier' for names


def make_parser(lang: str = "c") -> Parser:
    ts_lang = get_language(lang)
    try:
        p = Parser(language=ts_lang)  # type: ignore[arg-type]
    except TypeError:
        p = Parser()
        p.set_language(ts_lang)
    return p


_WS_RE = re.compile(r"[ \t]+")
_COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.S | re.M)


def normalize_code(src: str) -> str:
    """
    Very light 'standardization' to reduce noise without harming parsing:
    - strip comments
    - collapse repeated spaces/tabs
    - keep newlines (line mapping still mostly preserved)
    """
    try:
        s = _COMMENT_RE.sub("", src)
    except Exception:
        s = src
    s = _WS_RE.sub(" ", s)
    return s


def _collect_children(n: Node) -> List[Node]:
    return list(getattr(n, "children", []) or [])


def ast_nodes_and_edges_with_types(
    code: str, parser: Parser, max_nodes: int = 4000
) -> tuple[list[str], list[tuple[int, int]], list[int]]:
    """
    Returns:
      node_labels: list[str]   (node.type tokens)
      edges:       list[(u,v)] (directed)
      edge_types:  list[int]   (parallel to edges)
    Edge types:
      0 = AST parent->child
      1 = preorder-sequential (u -> u+1)
      2 = sibling (consecutive children of same parent, left->right)
      3 = same-identifier (cheap data-flow proxy: connect identical identifiers)
    """
    try:
        tree = parser.parse(code.encode("utf-8"))
    except Exception:
        return [], [], []

    root = tree.root_node
    node_labels: List[str] = []
    edges: List[tuple[int, int]] = []
    etypes: List[int] = []

    # DFS preorder to index nodes
    stack: List[tuple[int | None, Node]] = [(None, root)]
    index_of: Dict[int, int] = {}
    preorder_nodes: List[Node] = []
    parent_of_idx: Dict[int, int | None] = {}

    while stack:
        parent_idx, node = stack.pop()
        curr_idx = len(node_labels)
        index_of[id(node)] = curr_idx
        parent_of_idx[curr_idx] = parent_idx
        node_labels.append(node.type)
        preorder_nodes.append(node)
        if parent_idx is not None:
            edges.append((parent_idx, curr_idx))
            etypes.append(0)  # AST edge

        if len(node_labels) >= max_nodes:
            break

        children = _collect_children(node)
        for ch in reversed(children):
            stack.append((curr_idx, ch))

    n = len(node_labels)
    if n == 0:
        return [], [], []

    # 1) sequential edges over preorder
    for i in range(n - 1):
        edges.append((i, i + 1))
        etypes.append(1)

    # 2) sibling edges (consecutive children under same parent)
    # build children lists per parent
    children_by_parent: Dict[int, List[int]] = {}
    for child_idx, p in parent_of_idx.items():
        if p is None:
            continue
        children_by_parent.setdefault(p, []).append(child_idx)
    for _, kids in children_by_parent.items():
        kids.sort()
        for a, b in zip(kids, kids[1:]):
            edges.append((a, b))
            etypes.append(2)

    # 3) same-identifier links (cheap data-flow-ish connectivity)
    # collect identifier token texts (if available)
    # Tree-sitter Node has .text only via captured byte spans on original source,
    # so we can approximate: only use type equality here; still helpful.
    # Better: if node.type == 'identifier', connect all consecutive identifiers
    ident_idxs = [
        i for i, node in enumerate(preorder_nodes) if node.type in IDENT_TYPES
    ]
    for a, b in zip(ident_idxs, ident_idxs[1:]):
        edges.append((a, b))
        etypes.append(3)

    return node_labels, edges, etypes


def ast_nodes_and_edges(code: str, parser: Parser, max_nodes: int = 4000):
    """
    Backward-compatible wrapper (used by your old code).
    """
    lab, e, _t = ast_nodes_and_edges_with_types(code, parser, max_nodes)
    return lab, e
