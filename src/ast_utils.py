# src/ast_utils.py
from __future__ import annotations

from typing import List, Tuple

from tree_sitter import Node, Parser
from tree_sitter_languages import get_language


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


def ast_nodes_and_edges(
    code: str, parser: Parser, max_nodes: int = 4000
) -> tuple[list[str], list[tuple[int, int]]]:
    try:
        tree = parser.parse(code.encode("utf-8"))
    except Exception:
        return [], []

    root = tree.root_node
    node_types: List[str] = []
    edges: List[tuple[int, int]] = []
    stack: List[Tuple[int | None, Node]] = [(None, root)]

    while stack:
        parent_idx, node = stack.pop()
        curr_idx = len(node_types)
        node_types.append(node.type)
        if parent_idx is not None:
            edges.append((parent_idx, curr_idx))
        if len(node_types) >= max_nodes:
            break
        children: List[Node] = list(getattr(node, "children", []))
        for child in reversed(children):
            stack.append((curr_idx, child))
    return node_types, edges
