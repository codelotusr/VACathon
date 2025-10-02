# src/ast_utils.py
from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

from tree_sitter import Node, Parser
from tree_sitter_languages import get_language

# Common risky/sink APIs (extend as needed)
DEFAULT_SINK_PAT = re.compile(
    r"\b(strcpy|strcat|sprintf|vsprintf|gets|scanf|memcpy|memmove|strncpy|snprintf|recv|read|system|popen|exec|strlen)\b"
)

IDENT_TYPES: Set[str] = {"identifier"}
COND_TYPES: Set[str] = {
    "if_statement",
    "for_statement",
    "while_statement",
    "do_statement",
    "conditional_expression",
    "switch_statement",
}

_WS_RE = re.compile(r"[ \t]+")
_COMMENT_RE = re.compile(r"//.*?$|/\*.*?\*/", re.S | re.M)


def make_parser(lang: str = "c") -> Parser:
    """
    lang: "c" or "cpp". Handles both old/new tree-sitter Python APIs.
    """
    ts_lang = get_language(lang)
    try:
        p = Parser(language=ts_lang)  # type: ignore[arg-type]
    except TypeError:
        p = Parser()
        p.set_language(ts_lang)
    return p


def normalize_code(src: str) -> str:
    """
    Light standardization:
      - strip comments
      - collapse spaces/tabs (keep newlines)
    """
    try:
        s = _COMMENT_RE.sub("", src)
    except Exception:
        s = src
    s = _WS_RE.sub(" ", s)
    return s


def _collect_children(n: Node) -> List[Node]:
    return list(getattr(n, "children", []) or [])


def _parent_chain_contains_cond(
    idx: int, parent_of: Dict[int, Optional[int]], node_types: List[str]
) -> int:
    """Returns 1 if any ancestor is a conditional/loop/switch node."""
    p = parent_of.get(idx)
    while p is not None:
        if node_types[p] in COND_TYPES:
            return 1
        p = parent_of.get(p)
    return 0


def ast_nodes_and_edges(code: str, parser: Parser, max_nodes: int = 4000):
    """
    Compatibility shim returning (node_labels, edges) only.
    Used by the fast vocab pass.
    """
    labels, edges, _et, _parent = ast_nodes_and_edges_with_types(
        code, parser, max_nodes
    )
    return labels, edges


def ast_nodes_and_edges_with_types(
    code: str, parser: Parser, max_nodes: int = 4000
) -> tuple[list[str], list[tuple[int, int]], list[int], Dict[int, Optional[int]]]:
    """
    Returns:
      node_labels: list[str]   (node.type tokens)
      edges:       list[(u,v)] (directed)
      edge_types:  list[int]   parallel to edges
      parent_of:   mapping child_idx -> parent_idx

    Edge types:
      0 = AST parent->child
      1 = preorder sequential (u->u+1)
      2 = sibling (consecutive children under same parent)
      3 = same-identifier (cheap data-flow proxy)
    """
    try:
        tree = parser.parse(code.encode("utf-8"))
    except Exception:
        return [], [], [], {}

    root = tree.root_node
    node_labels: List[str] = []
    edges: List[tuple[int, int]] = []
    etypes: List[int] = []

    stack: List[tuple[Optional[int], Node]] = [(None, root)]
    parent_of_idx: Dict[int, Optional[int]] = {}

    preorder_nodes: List[Node] = []

    while stack:
        parent_idx, node = stack.pop()
        curr_idx = len(node_labels)
        parent_of_idx[curr_idx] = parent_idx
        node_labels.append(node.type)
        preorder_nodes.append(node)

        if parent_idx is not None:
            edges.append((parent_idx, curr_idx))
            etypes.append(0)  # AST

        if len(node_labels) >= max_nodes:
            break

        children = _collect_children(node)
        for ch in reversed(children):
            stack.append((curr_idx, ch))

    n = len(node_labels)
    if n == 0:
        return [], [], [], {}

    # 1) sequential (preorder)
    for i in range(n - 1):
        edges.append((i, i + 1))
        etypes.append(1)

    # 2) sibling edges
    kids_by_parent: Dict[int, List[int]] = {}
    for child_idx, p in parent_of_idx.items():
        if p is None:
            continue
        kids_by_parent.setdefault(p, []).append(child_idx)
    for _, kids in kids_by_parent.items():
        kids.sort()
        for a, b in zip(kids, kids[1:]):
            edges.append((a, b))
            etypes.append(2)

    # 3) same-identifier links (approximate)
    ident_idxs = [
        i for i, node in enumerate(preorder_nodes) if node.type in IDENT_TYPES
    ]
    for a, b in zip(ident_idxs, ident_idxs[1:]):
        edges.append((a, b))
        etypes.append(3)

    return node_labels, edges, etypes, parent_of_idx


def detect_sink_lines(
    code: str, sink_pat: re.Pattern[str] = DEFAULT_SINK_PAT
) -> List[int]:
    """
    Return 0-based line numbers that contain a risky/sink API name.
    """
    lines = code.splitlines()
    out = []
    for i, ln in enumerate(lines):
        if sink_pat.search(ln):
            out.append(i)
    return out


def k_hop_slice_mask(
    n_nodes: int,
    node_to_line: Dict[int, int],
    start_lines: List[int],
    k_hops: int,
    edges: List[tuple[int, int]],
) -> List[bool]:
    """
    Keep nodes within k preorder-hops (undirected) from any node mapped to sink lines.
    Preorder index is already a strong locality signal; this is a pragmatic approximation.

    Returns boolean mask length n_nodes.
    """
    if n_nodes == 0 or not start_lines:
        return [True] * n_nodes

    # Map sink lines -> candidate node indices
    seeds: List[int] = [
        idx for idx, ln in node_to_line.items() if ln in set(start_lines)
    ]
    if not seeds:
        return [True] * n_nodes

    # undirected adjacency on preorder indices
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for a, b in edges:
        if 0 <= a < n_nodes and 0 <= b < n_nodes:
            adj[a].append(b)
            adj[b].append(a)

    from collections import deque

    dist = [-1] * n_nodes
    q = deque()
    for s in seeds:
        dist[s] = 0
        q.append(s)
    while q:
        u = q.popleft()
        if dist[u] >= k_hops:
            continue
        for v in adj[u]:
            if dist[v] == -1:
                dist[v] = dist[u] + 1
                q.append(v)
    return [d != -1 and d <= k_hops for d in dist]
