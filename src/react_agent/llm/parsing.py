"""Parse free-form model text into a validated ModelStep."""

from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from react_agent.types import ModelStep


class ParseError(ValueError):
    """The model output could not be turned into a ModelStep."""


def _extract_json_object(text: str) -> dict[str, Any]:
    """Return the first top-level JSON object in `text` (tolerates code fences/prose)."""
    decoder = json.JSONDecoder()
    start = text.find("{")
    while start != -1:
        try:
            obj, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict):
            return obj
        start = text.find("{", start + 1)
    raise ParseError("no JSON object found in model output")


def parse_model_step(text: str) -> ModelStep:
    """Parse model output into a ModelStep or raise ParseError."""
    data = _extract_json_object(text)
    try:
        return ModelStep.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        raise ParseError(f"invalid step: {first['msg']}") from exc
