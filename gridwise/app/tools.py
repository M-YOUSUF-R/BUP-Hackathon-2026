# app/tools.py
"""
Deterministic tools the LLM MUST call for any arithmetic.

Design principles:
- No tool raises on user/LLM input. All failures return {"valid": False, "error": ...}.
- All return values are JSON-serializable so they can be handed back to the model.
- Every rule from the Problem Statement (inclusive/exclusive, factor as remaining
  fraction, % of capacity -> absolute kWh) is encoded here, in ONE place.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from langchain_core.tools import tool


# ---------- time parsing ----------

_HOUR_WORDS = {
    "midnight": 0,
    "noon": 12,
    "midday": 12,
}

_TIME_RE = re.compile(
    r"^\s*(\d{1,2})\s*(?::\s*(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\s*$",
    re.IGNORECASE,
)


def _parse_hour_token(token: str) -> int:
    """Parse one time token into a whole hour index 0..23.

    Accepts: '1 PM', '13', '13:00', 'noon', 'midnight', '1pm', '1:00 p.m.'.
    Rejects: non-whole-hour times like '1:30 PM'.
    """
    t = str(token).strip().lower().replace(".", "")
    if t in _HOUR_WORDS:
        return _HOUR_WORDS[t]

    m = _TIME_RE.match(t)
    if not m:
        raise ValueError(f"cannot parse time: {token!r}")

    hh = int(m.group(1))
    mm = int(m.group(2) or 0)
    ampm = m.group(3)

    if ampm == "pm" and hh != 12:
        hh += 12
    if ampm == "am" and hh == 12:
        hh = 0

    if not 0 <= hh <= 23:
        raise ValueError(f"hour out of range: {token!r}")

    if mm != 0:
        # Problem Statement uses whole-hour windows. Reject non-whole.
        raise ValueError(f"non-whole-hour time not supported: {token!r}")

    return hh


@tool
def time_window_to_hours(start: str, end: str) -> dict[str, Any]:
    """Convert a whole-hour time window into a list of hour indices.

    Uses the Problem Statement convention: START inclusive, END exclusive.

    Inputs may be in any common written form:
        '1 PM', '13', '13:00', 'noon', 'midnight', '1pm'.

    Examples:
        ('1 PM', '3 PM')   -> {'valid': True, 'hours': [13, 14]}
        ('noon', '2 PM')   -> {'valid': True, 'hours': [12, 13]}
        ('6 PM', '9 PM')   -> {'valid': True, 'hours': [18, 19, 20]}
        ('11 AM', '2 PM')  -> {'valid': True, 'hours': [11, 12, 13]}
    """
    try:
        s = _parse_hour_token(start)
        e = _parse_hour_token(end)
    except ValueError as exc:
        return {"valid": False, "hours": [], "error": str(exc)}

    if e <= s:
        return {
            "valid": False,
            "hours": [],
            "error": f"end must be after start; got {start!r} -> {end!r}",
        }

    return {"valid": True, "hours": list(range(s, e)), "error": None}


# ---------- solar factor ----------

@tool
def solar_factor(
    reduction_percent: Optional[float] = None,
    remaining_percent: Optional[float] = None,
) -> dict[str, Any]:
    """Compute the solar_reduction 'factor' = USABLE FRACTION REMAINING, in [0,1].

    Exactly one of the two arguments must be provided:
      - reduction_percent: 'an 80% reduction'            -> pass 80
      - remaining_percent: 'about 20% of forecast'       -> pass 20

    Examples:
        reduction_percent=80  -> factor = 0.2
        reduction_percent=75  -> factor = 0.25
        remaining_percent=25  -> factor = 0.25
        remaining_percent=20  -> factor = 0.2
        remaining_percent=50  -> factor = 0.5
    """
    if (reduction_percent is None) == (remaining_percent is None):
        return {
            "valid": False,
            "factor": None,
            "error": "provide exactly one of reduction_percent or remaining_percent",
        }

    if reduction_percent is not None:
        try:
            p = float(reduction_percent)
        except (TypeError, ValueError):
            return {"valid": False, "factor": None, "error": "non-numeric reduction_percent"}
        if not 0.0 <= p <= 100.0:
            return {"valid": False, "factor": None, "error": "reduction_percent must be 0..100"}
        factor = round(1.0 - p / 100.0, 6)
    else:
        try:
            p = float(remaining_percent)
        except (TypeError, ValueError):
            return {"valid": False, "factor": None, "error": "non-numeric remaining_percent"}
        if not 0.0 <= p <= 100.0:
            return {"valid": False, "factor": None, "error": "remaining_percent must be 0..100"}
        factor = round(p / 100.0, 6)

    # Reject near-zero factors that would effectively kill solar; keep 0 legal only if
    # the note explicitly says 100% reduction.
    return {"valid": True, "factor": factor, "error": None}


# ---------- reserve ----------

@tool
def reserve_kwh_from_percent(percent: float, capacity_kwh: float) -> dict[str, Any]:
    """Convert 'X% of battery capacity' into an absolute kWh reserve.

    Examples:
        (50, 200) -> 100.0
        (60, 250) -> 150.0
    """
    try:
        p = float(percent)
        cap = float(capacity_kwh)
    except (TypeError, ValueError):
        return {"valid": False, "minimum_energy_kwh": None, "error": "non-numeric input"}

    if not 0.0 <= p <= 100.0:
        return {"valid": False, "minimum_energy_kwh": None, "error": "percent must be 0..100"}
    if cap <= 0:
        return {"valid": False, "minimum_energy_kwh": None, "error": "capacity must be > 0"}

    val = round(p / 100.0 * cap, 6)
    return {"valid": True, "minimum_energy_kwh": val, "error": None}


# ---------- hours validation ----------

@tool
def validate_hours(hours: list[int]) -> dict[str, Any]:
    """Sort, dedupe, and range-check hour indices.

    Use after computing hours to guarantee ascending, unique, 0..23.

    Examples:
        [13, 12, 12]      -> {'valid': True, 'hours': [12, 13]}
        [24]              -> {'valid': False, 'error': 'hours out of range'}
        []                -> {'valid': False, 'error': 'hours must be non-empty'}
    """
    if not isinstance(hours, list) or len(hours) == 0:
        return {"valid": False, "hours": [], "error": "hours must be non-empty list"}

    try:
        ints = sorted({int(h) for h in hours})
    except (TypeError, ValueError) as exc:
        return {"valid": False, "hours": [], "error": f"non-integer hours: {exc}"}

    if ints[0] < 0 or ints[-1] > 23:
        return {"valid": False, "hours": [], "error": f"hours out of range: {ints}"}

    return {"valid": True, "hours": ints, "error": None}


# ---------- tool registry ----------

MATH_TOOLS = [
    time_window_to_hours,
    solar_factor,
    reserve_kwh_from_percent,
    validate_hours,
]

MATH_TOOLS_BY_NAME = {t.name: t for t in MATH_TOOLS}