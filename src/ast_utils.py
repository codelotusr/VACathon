from __future__ import annotations

from tree_sitter import Parser
from tree_sitter_languages import get_language


def make_c_parser() -> Parser:
    lang = get_language("c")
    return Parser(language=lang)
