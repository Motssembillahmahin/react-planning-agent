"""Small text utilities shared by the critic and the reflection step."""

from __future__ import annotations

import re

NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?")
_WORD = re.compile(r"[a-z][a-z']*")
_FILLER_TEXT = (
    "the a an and or of to in on at by for from with is are was were be been it its this that "
    "about approximately roughly around nearly almost some times than per answer result "
    "which who what when where there their they has have had high tall long deep"
)
FILLER = frozenset(_FILLER_TEXT.split())


def numbers(text: str) -> list[str]:
    """Numbers in `text` as strings, thousands separators removed."""
    return [m.replace(",", "") for m in NUMBER.findall(text)]


def stem(word: str) -> str:
    word = word.removesuffix("'s").strip("'")
    for suffix in ("ing", "ed", "es", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)]
    return word


def content_words(text: str) -> set[str]:
    """Lower-cased, crudely stemmed words of 3+ letters, minus filler words."""
    words = (stem(w) for w in _WORD.findall(text.lower()))
    return {w for w in words if len(w) >= 3 and w not in FILLER}


def info_tokens(text: str) -> set[str]:
    """Content words plus normalised numbers: the units of 'information' for reflection."""
    return content_words(text) | {str(float(n)) for n in numbers(text)}
