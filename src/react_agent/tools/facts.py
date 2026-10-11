"""An in-memory key-value fact store tool (get / set / list)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from react_agent.tools.base import Tool, ToolInputError

DEFAULT_FACTS: dict[str, str] = {
    "company_founded": "2019",
    "office_city": "Lisbon",
    "team_size": "12",
    "project_codename": "Heron",
    "release_version": "2.4.1",
    "budget_eur": "48000",
}


class FactArgs(BaseModel):
    op: Literal["get", "set", "list"] = "get"
    key: str | None = None
    value: str | None = None


class FactStore:
    def __init__(self, facts: dict[str, str] | None = None) -> None:
        self.facts = dict(DEFAULT_FACTS if facts is None else facts)

    def __call__(self, args: FactArgs) -> str:
        if args.op == "list":
            return "keys: " + (", ".join(sorted(self.facts)) or "(empty)")
        if not args.key:
            raise ToolInputError(f"'{args.op}' requires a key")
        key = args.key.strip().lower()
        if args.op == "set":
            if args.value is None:
                raise ToolInputError("'set' requires a value")
            self.facts[key] = args.value
            return f"stored {key} = {args.value}"
        if key in self.facts:
            return f"{key} = {self.facts[key]}"
        return f"No fact stored for '{key}'."


def fact_store_tool(store: FactStore | None = None, name: str = "facts") -> Tool:
    return Tool(
        name=name,
        description="Key-value fact store. op='get'|'set'|'list', key, value.",
        args_model=FactArgs,
        fn=store or FactStore(),
    )
