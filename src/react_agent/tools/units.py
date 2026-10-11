"""Unit converter for length, mass, time, volume, speed, and temperature."""

from __future__ import annotations

from pydantic import BaseModel

from react_agent.tools.base import Tool, ToolInputError
from react_agent.tools.calculator import format_number

# Factor to the dimension's base unit.
_LINEAR: dict[str, dict[str, float]] = {
    "length": {
        "m": 1.0,
        "km": 1000.0,
        "cm": 0.01,
        "mm": 0.001,
        "mi": 1609.344,
        "ft": 0.3048,
        "in": 0.0254,
        "yd": 0.9144,
    },
    "mass": {
        "kg": 1.0,
        "g": 0.001,
        "mg": 1e-6,
        "t": 1000.0,
        "lb": 0.45359237,
        "oz": 0.028349523125,
    },
    "time": {"s": 1.0, "min": 60.0, "h": 3600.0, "day": 86400.0, "week": 604800.0},
    "volume": {"l": 1.0, "ml": 0.001, "m3": 1000.0, "gal": 3.785411784},
    "speed": {"m/s": 1.0, "km/h": 1 / 3.6, "mph": 0.44704},
}
_TEMPERATURE = {"c", "f", "k"}

_ALIASES: dict[str, str] = {
    "meter": "m", "meters": "m", "metre": "m", "metres": "m",
    "kilometer": "km", "kilometers": "km", "kilometre": "km", "kilometres": "km",
    "centimeter": "cm", "centimeters": "cm", "millimeter": "mm", "millimeters": "mm",
    "mile": "mi", "miles": "mi", "foot": "ft", "feet": "ft",
    "inch": "in", "inches": "in", "yard": "yd", "yards": "yd",
    "kilogram": "kg", "kilograms": "kg", "gram": "g", "grams": "g",
    "pound": "lb", "pounds": "lb", "lbs": "lb", "ounce": "oz", "ounces": "oz",
    "tonne": "t", "tonnes": "t",
    "second": "s", "seconds": "s", "sec": "s", "minute": "min", "minutes": "min",
    "hour": "h", "hours": "h", "hr": "h", "days": "day", "weeks": "week",
    "liter": "l", "liters": "l", "litre": "l", "litres": "l",
    "milliliter": "ml", "milliliters": "ml", "gallon": "gal", "gallons": "gal",
    "kph": "km/h", "kmh": "km/h",
    "celsius": "c", "°c": "c", "fahrenheit": "f", "°f": "f", "kelvin": "k",
}  # fmt: skip


def _normalize(unit: str) -> str:
    u = unit.strip().lower()
    return _ALIASES.get(u, u)


def _dimension(unit: str) -> str:
    if unit in _TEMPERATURE:
        return "temperature"
    for dim, table in _LINEAR.items():
        if unit in table:
            return dim
    raise ToolInputError(f"unknown unit '{unit}'")


def _to_kelvin(value: float, unit: str) -> float:
    if unit == "c":
        return value + 273.15
    if unit == "f":
        return (value - 32) * 5 / 9 + 273.15
    return value


def _from_kelvin(value: float, unit: str) -> float:
    if unit == "c":
        return value - 273.15
    if unit == "f":
        return (value - 273.15) * 9 / 5 + 32
    return value


def convert(value: float, from_unit: str, to_unit: str) -> float:
    src, dst = _normalize(from_unit), _normalize(to_unit)
    dim_src, dim_dst = _dimension(src), _dimension(dst)
    if dim_src != dim_dst:
        raise ToolInputError(f"cannot convert {dim_src} ({from_unit}) to {dim_dst} ({to_unit})")
    if dim_src == "temperature":
        result = _from_kelvin(_to_kelvin(value, src), dst)
    else:
        table = _LINEAR[dim_src]
        result = value * table[src] / table[dst]
    return round(result, 9)


class UnitArgs(BaseModel):
    value: float
    from_unit: str
    to_unit: str


def _run(args: UnitArgs) -> str:
    result = convert(args.value, args.from_unit, args.to_unit)
    return f"{format_number(args.value)} {args.from_unit} = {format_number(result)} {args.to_unit}"


def unit_converter_tool(name: str = "unit_converter") -> Tool:
    return Tool(
        name=name,
        description="Convert a value between units (length, mass, time, volume, speed, C/F/K).",
        args_model=UnitArgs,
        fn=_run,
    )
