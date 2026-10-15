"""Fixed CFG rules from legacy/AutomataSentenceValidator.py + lexicon-driven terminals.

The production rules are unchanged from the original grammar checker; only the
terminal word lists (categories a-o) come from the lexicon.
"""
from __future__ import annotations

import difflib
import re

from nltk import CFG, ChartParser, Nonterminal, Production, Tree

from ..helpers.lexicon import normalize

# Fixed structural rules (based on legacy/AutomataSentenceValidator.py)
RULES: dict[str, list[list[str]]] = {
    "S": [["D"], ["E"], ["F"], ["O"]],
    "A": [["a", "N"]],
    "B": [["b", "N"]],
    "C": [["c", "N"]],
    "D": [["d", "G"], ["d", "K"], ["d", "L"], ["d", "M"], ["d", "C"]],
    "E": [["e", "J"], ["e", "K"], ["e", "L"], ["e", "M"], ["e", "C"]],
    "F": [["f", "H"], ["f", "K"], ["f", "L"], ["f", "M"], ["f", "C"]],
    "G": [["g", "K"], ["g", "L"], ["g", "M"]],
    "H": [["h", "K"], ["h", "L"], ["h", "M"]],
    "I": [["i", "K"], ["i", "L"], ["i", "M"]],
    "J": [["j", "I"]],
    "K": [["k", "D"], ["k", "E"], ["k", "F"], ["k", "O"]],
    "L": [["l", "A"]],
    "M": [["m", "B"]],
    "N": [["n"]],
    "O": [["o", "K"], ["o", "L"], ["o", "M"], ["o", "C"]],
}

# Builder templates: each is a valid derivation skeleton (one letter per slot)
TEMPLATES: list[tuple[str, list[str]]] = [
    ("Adjective + si + name", ["o", "m", "b", "n"]),
    ("Adjective + ang + noun", ["o", "l", "a", "n"]),
    ("Adjective + pronoun", ["o", "c", "n"]),
    ("Adjective + kag + adjective + si + name", ["o", "k", "o", "m", "b", "n"]),
    ("Past + ang + noun", ["d", "l", "a", "n"]),
    ("Past + si + name", ["d", "m", "b", "n"]),
    ("Past + pronoun", ["d", "c", "n"]),
    ("Past + time + ang + noun", ["d", "g", "l", "a", "n"]),
    ("Past + time + si + name", ["d", "g", "m", "b", "n"]),
    ("Present + ang + noun", ["e", "l", "a", "n"]),
    ("Present + si + name", ["e", "m", "b", "n"]),
    ("Present + pronoun", ["e", "c", "n"]),
    ("Present + sang + manner + ang + noun", ["e", "j", "i", "l", "a", "n"]),
    ("Future + ang + noun", ["f", "l", "a", "n"]),
    ("Future + si + name", ["f", "m", "b", "n"]),
    ("Future + pronoun", ["f", "c", "n"]),
    ("Future + time + ang + noun", ["f", "h", "l", "a", "n"]),
    ("Past + kag + adjective + si + name", ["d", "k", "o", "m", "b", "n"]),
]

_TOKEN_RE = re.compile(r"[a-z][a-z'-]*|\d+|[^ \s]")

# Plain-English names for every grammar symbol (used in trees and explanations)
PLAIN_LABELS: dict[str, str] = {
    "S": "Sentence",
    "D": "Past-tense sentence",
    "E": "Present-tense sentence",
    "F": "Future-tense sentence",
    "O": "Describing sentence",
    "A": "Common noun + end mark",
    "B": "Name + end mark",
    "C": "Pronoun + end mark",
    "G": "Past-time phrase",
    "H": "Now/future-time phrase",
    "I": "Manner phrase",
    "J": "'sang' + manner",
    "K": "'kag' + clause",
    "L": "'ang' + noun",
    "M": "'si'/'kay' + name",
    "N": "End mark (.)",
    "a": "common noun",
    "b": "name (proper noun)",
    "c": "pronoun",
    "d": "past-tense verb",
    "e": "present-tense verb",
    "f": "future-tense verb",
    "g": "past-time word",
    "h": "now/future-time word",
    "i": "manner adverb",
    "j": "'sang' (linker)",
    "k": "'kag' (and)",
    "l": "'ang' (topic marker)",
    "m": "'si'/'kay' (name marker)",
    "n": "period",
    "o": "adjective",
}


def tokenize(sentence: str) -> list[str]:
    """Split on whitespace, peeling trailing/leading punctuation, accents stripped."""
    return _TOKEN_RE.findall(normalize(sentence))


class GrammarChecker:
    def __init__(self, categories: dict):
        self.categories = categories
        self.vocab: dict[str, set[str]] = {
            letter: set(cat.words) for letter, cat in categories.items()
        }
        self.all_vocab = sorted(set().union(*self.vocab.values()))
        self.grammar = self._build_grammar()
        self.parser = ChartParser(self.grammar)

    def _build_grammar(self) -> CFG:
        productions = [
            Production(Nonterminal(lhs), [Nonterminal(sym) for sym in rhs])
            for lhs, rhss in RULES.items()
            for rhs in rhss
        ]
        for letter, cat in self.categories.items():
            nt = Nonterminal(letter)
            productions.extend(Production(nt, [word]) for word in cat.words)
        return CFG(Nonterminal("S"), productions)

    def check(self, sentence: str) -> dict:
        tokens = tokenize(sentence)
        try:
            trees = list(self.parser.parse(tokens))
        except ValueError:  # nltk raises when tokens are outside the grammar
            trees = []
        unknown = {}
        for tok in tokens:
            if not any(tok in words for words in self.vocab.values()):
                unknown[tok] = difflib.get_close_matches(tok, self.all_vocab, n=3, cutoff=0.6)
        return {
            "sentence": sentence,
            "tokens": tokens,
            "valid": bool(trees),
            "trees": trees,
            "unknown": unknown,
        }


def sentence_kind(tree: Tree) -> str:
    """Plain-English name of the sentence type (e.g. 'Past-tense sentence')."""
    top = tree[0].label() if tree and isinstance(tree[0], Tree) else tree.label()
    return PLAIN_LABELS.get(top, top)


def leaf_roles(tree: Tree) -> list[tuple[str, str]]:
    """[(word, its role in the sentence), ...] for every word in the parse."""
    roles: list[tuple[str, str]] = []

    def walk(node, parent_raw: str | None = None) -> None:
        if isinstance(node, Tree):
            for child in node:
                walk(child, node.label())
        else:
            plain = PLAIN_LABELS.get(parent_raw or "", parent_raw or "")
            roles.append((str(node), plain))

    walk(tree)
    return roles


def tree_to_dot(tree: Tree, plain: bool = True, dark: bool = False) -> str:
    """Convert an nltk parse tree to Graphviz DOT (rendered by st.graphviz_chart).

    plain=True shows English labels ("Past-tense sentence"); plain=False shows
    the original grammar symbols (S, D, a ...).
    """
    node_fill = "#29312b" if dark else "#f1f0eb"
    leaf_fill = "#34483a" if dark else "#e3ece5"
    border = "#758579" if dark else "#778278"
    text = "#e8ebe7" if dark else "#292b27"
    lines = [
        "digraph ParseTree {",
        f'  node [shape=box, style="filled", fillcolor="{node_fill}", color="{border}", fontcolor="{text}", fontname="Helvetica", fontsize=12, margin="0.14,0.08"];',
        f'  edge [color="{border}", arrowsize=0.7];',
    ]
    counter = {"n": 0}

    def walk(node, parent_id: int | None = None) -> None:
        nid = counter["n"]
        counter["n"] += 1
        is_leaf = not isinstance(node, Tree)
        if is_leaf:
            label = str(node)
        else:
            raw = node.label()
            label = PLAIN_LABELS.get(raw, raw) if plain else raw
        label = label.replace('"', '\\"')
        fill = f', fillcolor="{leaf_fill}"' if is_leaf else ""
        lines.append(f'  n{nid} [label="{label}"{fill}];')
        if parent_id is not None:
            lines.append(f"  n{parent_id} -> n{nid};")
        if isinstance(node, Tree):
            for child in node:
                walk(child, nid)

    walk(tree)
    lines.append("}")
    return "\n".join(lines)


def result_summary(result: dict) -> str:
    n = len(result["trees"])
    if result["valid"]:
        return f"This sentence follows the grammar rules ({n} reading{'s' if n != 1 else ''})."
    return "This sentence doesn't match any grammar pattern yet."
