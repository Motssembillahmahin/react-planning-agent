"""Keyword search over a small bundled text corpus (TF-IDF style scoring)."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from functools import cache
from importlib import resources

from pydantic import BaseModel, Field

from react_agent.tools.base import Tool

_STOPWORDS_TEXT = (
    "a an and are as at be by for from how in is it its of on or the to was what when where "
    "which who why with does did many much tall high long far"
)
_STOPWORDS = frozenset(_STOPWORDS_TEXT.split())
_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


@dataclass(frozen=True)
class Document:
    id: str
    title: str
    text: str

    def render(self) -> str:
        return f"[{self.id}] {self.title}: {self.text}"


@cache
def load_corpus() -> tuple[Document, ...]:
    raw = resources.files("react_agent.data").joinpath("corpus.json").read_text(encoding="utf-8")
    return tuple(Document(**d) for d in json.loads(raw))


def search(query: str, k: int = 2, corpus: tuple[Document, ...] | None = None) -> list[Document]:
    docs = corpus if corpus is not None else load_corpus()
    terms = set(tokenize(query))
    if not terms:
        return []
    doc_tokens = [tokenize(f"{d.title} {d.text}") for d in docs]
    n = len(docs)
    scored: list[tuple[float, int]] = []
    for i, tokens in enumerate(doc_tokens):
        score = 0.0
        for term in terms:
            tf = tokens.count(term)
            if tf:
                df = sum(1 for t in doc_tokens if term in t)
                score += tf * math.log(1 + n / df)
        if score > 0:
            scored.append((score, i))
    scored.sort(key=lambda s: (-s[0], s[1]))
    return [docs[i] for _, i in scored[:k]]


class SearchArgs(BaseModel):
    query: str = Field(min_length=1)
    k: int = Field(default=2, ge=1, le=5)


def _run(args: SearchArgs) -> str:
    hits = search(args.query, args.k)
    if not hits:
        return f"No results for '{args.query}'."
    return "\n".join(d.render() for d in hits)


def search_tool(name: str = "search") -> Tool:
    return Tool(
        name=name,
        description="Search a small local encyclopedia; returns the top matching passages.",
        args_model=SearchArgs,
        fn=_run,
    )
