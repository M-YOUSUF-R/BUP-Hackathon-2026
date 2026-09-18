# app/guardrails.py
from typing import Any
from .schemas import RawDirective, RawInterpretation, DirectiveInterpretation


class GuardrailError(Exception):
    """Raised when a directive cannot be safely repaired."""


ALLOWED_TYPES = {
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
}


def _clean_hours(hours: list[int] | None) -> list[int]:
    """Sort, dedupe, and range-check hours. Returns [] if input is None/empty."""
    if not hours:
        return []
    unique = sorted(set(int(h) for h in hours))
    if unique[0] < 0 or unique[-1] > 23:
        raise GuardrailError(f"hours out of range: {unique}")
    return unique


def _validate_one(
    raw: RawDirective, capacity_kwh: float
) -> DirectiveInterpretation:
    """Validate and normalize a single directive. Repairs safe issues."""
    if raw.directive_type not in ALLOWED_TYPES:
        raise GuardrailError(f"unsupported directive_type: {raw.directive_type}")

    hours = _clean_hours(raw.hours)

    # -------- no_op --------
    if raw.directive_type == "no_op":
        if raw.applies is not False:
            raise GuardrailError("no_op must have applies=false")
        return DirectiveInterpretation(
            note_index=raw.note_index,
            applies=False,
            directive_type="no_op",
            structured_adjustment=None,
            explanation=raw.explanation or "No effect on today's schedule.",
        )

    # -------- all real directives require applies=true --------
    if raw.applies is not True:
        raise GuardrailError(
            f"{raw.directive_type} must have applies=true"
        )
    if not hours:
        raise GuardrailError(f"{raw.directive_type} requires non-empty hours")

    # -------- type-specific shape checks --------
    if raw.directive_type == "solar_reduction":
        if raw.factor is None or not (0.0 <= float(raw.factor) <= 1.0):
            raise GuardrailError("solar_reduction requires factor in [0,1]")
        adj: dict[str, Any] = {"hours": hours, "factor": float(raw.factor)}

    elif raw.directive_type == "minimum_battery_reserve":
        if raw.minimum_energy_kwh is None or raw.minimum_energy_kwh < 0:
            raise GuardrailError(
                "minimum_battery_reserve requires non-negative minimum_energy_kwh"
            )
        # Clamp above capacity — reserve cannot exceed physical capacity.
        val = min(float(raw.minimum_energy_kwh), float(capacity_kwh))
        adj = {"hours": hours, "minimum_energy_kwh": val}

    elif raw.directive_type in ("no_charge_window", "no_discharge_window"):
        adj = {"hours": hours}

    elif raw.directive_type == "max_grid_window":
        if raw.max_grid_kwh is None or raw.max_grid_kwh < 0:
            raise GuardrailError(
                "max_grid_window requires non-negative max_grid_kwh"
            )
        adj = {"hours": hours, "max_grid_kwh": float(raw.max_grid_kwh)}

    else:  # pragma: no cover — should be unreachable
        raise GuardrailError(f"unhandled directive: {raw.directive_type}")

    return DirectiveInterpretation(
        note_index=raw.note_index,
        applies=True,
        directive_type=raw.directive_type,
        structured_adjustment=adj,
        explanation=raw.explanation or "",
    )


def validate_interpretation(
    raw: RawInterpretation,
    n_notes: int,
    capacity_kwh: float,
) -> list[DirectiveInterpretation]:
    """Validate the full LLM output.

    Guarantees:
    - Exactly one entry per note, in note_index order.
    - Each entry satisfies the shape rules for its directive_type.
    - A note that fails validation is downgraded to no_op (never dropped).
    """
    by_index: dict[int, RawDirective] = {}
    for d in raw.directives:
        if d.note_index in by_index:
            # duplicate index — keep the first, mark the second as invalid
            continue
        by_index[d.note_index] = d

    result: list[DirectiveInterpretation] = []
    for i in range(n_notes):
        raw_d = by_index.get(i)
        if raw_d is None:
            # missing note — emit safe no_op
            result.append(
                DirectiveInterpretation(
                    note_index=i,
                    applies=False,
                    directive_type="no_op",
                    structured_adjustment=None,
                    explanation="Note was not interpreted; treated as no_op.",
                )
            )
            continue

        try:
            result.append(_validate_one(raw_d, capacity_kwh))
        except GuardrailError as exc:
            # Safe downgrade rather than failing the whole request.
            result.append(
                DirectiveInterpretation(
                    note_index=i,
                    applies=False,
                    directive_type="no_op",
                    structured_adjustment=None,
                    explanation=f"Invalid LLM output downgraded to no_op: {exc}",
                )
            )

    return result
