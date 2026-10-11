"""A safe calculator: parses arithmetic with `ast` and walks a whitelist. Never uses eval()."""

from __future__ import annotations

import ast
import math
import operator
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from react_agent.tools.base import Tool, ToolInputError

Number = int | float

_BINOPS: dict[type[ast.operator], Callable[[Any, Any], Any]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[Any], Any]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}
_FUNCS: dict[str, Callable[..., Any]] = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "floor": math.floor,
    "ceil": math.ceil,
}
_CONSTS: dict[str, float] = {"pi": math.pi, "e": math.e}

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 1000
MAX_INT_BITS = 4096


def _check_number(value: Any) -> Number:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ToolInputError("result is not a number")
    if isinstance(value, int) and value.bit_length() > MAX_INT_BITS:
        raise ToolInputError("result too large")
    if isinstance(value, float) and not math.isfinite(value):
        raise ToolInputError("result is not finite")
    return value


def _eval(node: ast.AST) -> Number:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant):
        return _check_number(node.value)
    if isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolInputError("exponent too large")
        return _check_number(_BINOPS[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _check_number(_UNARY[type(node.op)](_eval(node.operand)))
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        args = [_eval(a) for a in node.args]
        return _check_number(_FUNCS[node.func.id](*args))
    raise ToolInputError(f"unsupported expression element: {type(node).__name__}")


def evaluate(expression: str) -> Number:
    """Evaluate an arithmetic expression safely. Raises ToolInputError on bad input."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ToolInputError(f"expression longer than {MAX_EXPRESSION_LENGTH} characters")
    expression = expression.replace("^", "**").replace("\u00d7", "*").replace("\u00f7", "/")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
    except SyntaxError as exc:
        raise ToolInputError(f"syntax error: {exc.msg}") from exc
    try:
        return _eval(tree)
    except ToolInputError:
        raise
    except (ArithmeticError, ValueError, TypeError) as exc:
        raise ToolInputError(f"math error: {exc}") from exc


def format_number(value: Number) -> str:
    """Render integers without a trailing '.0' and floats with 10 significant digits."""
    if isinstance(value, float) and value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.10g}"
    return str(value)


class CalculatorArgs(BaseModel):
    expression: str = Field(description="arithmetic expression, e.g. '17 * 23 + 4'")


def _run(args: CalculatorArgs) -> str:
    return format_number(evaluate(args.expression))


def calculator_tool(name: str = "calculator") -> Tool:
    return Tool(
        name=name,
        description="Evaluate an arithmetic expression (+ - * / // % ** sqrt log sin ...).",
        args_model=CalculatorArgs,
        fn=_run,
    )
