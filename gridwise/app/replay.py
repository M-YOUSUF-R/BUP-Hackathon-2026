# app/replay.py
"""Re-derive totals and validate a schedule against the request.

This mirrors what the judge does. Run it on your own response before submitting.
"""
from .schemas import (
    HourInput, BatteryInput, DirectiveInterpretation, HourlyPlanEntry,
)
from .optimizer import _apply_directives


TOL = 0.01


def replay_and_check(
    hours: list[HourInput],
    battery: BatteryInput,
    directives: list[DirectiveInterpretation],
    plan: list[HourlyPlanEntry],
) -> list[str]:
    """Return a list of error strings. Empty list means schedule is valid."""
    errors: list[str] = []
    eff_solar, active_min, no_charge_h, no_discharge_h, grid_cap = (
        _apply_directives(hours, battery, directives)
    )

    if len(plan) != 24:
        errors.append(f"plan length {len(plan)} != 24")
        return errors

    prev_energy = battery.initial_energy_kwh

    for p in plan:
        h = p.hour
        h_in = hours[h]

        # 1. Energy balance
        ch = p.battery_kwh if p.battery_action == "charge" else 0.0
        dis = p.battery_kwh if p.battery_action == "discharge" else 0.0
        lhs = p.grid_kwh + p.solar_used_kwh + dis
        rhs = h_in.demand_kwh + ch
        if abs(lhs - rhs) > TOL:
            errors.append(f"hour {h}: balance {lhs} != {rhs}")

        # 2. Solar cap
        if p.solar_used_kwh > eff_solar[h] + TOL:
            errors.append(
                f"hour {h}: solar_used {p.solar_used_kwh} > eff {eff_solar[h]}"
            )

        # 3. Rate limits
        if p.battery_action == "charge" and p.battery_kwh > battery.max_charge_kwh_per_hour + TOL:
            errors.append(f"hour {h}: charge rate exceeded")
        if p.battery_action == "discharge" and p.battery_kwh > battery.max_discharge_kwh_per_hour + TOL:
            errors.append(f"hour {h}: discharge rate exceeded")

        # 4. Directive bans / caps
        if h in no_charge_h and ch > TOL:
            errors.append(f"hour {h}: charging in no_charge_window")
        if h in no_discharge_h and dis > TOL:
            errors.append(f"hour {h}: discharging in no_discharge_window")
        if grid_cap[h] is not None and p.grid_kwh > grid_cap[h] + TOL:
            errors.append(
                f"hour {h}: grid {p.grid_kwh} > cap {grid_cap[h]}"
            )

        # 5. Battery state consistency
        expected_after = prev_energy + ch - dis
        if abs(p.battery_energy_after_kwh - expected_after) > TOL:
            errors.append(
                f"hour {h}: E_after {p.battery_energy_after_kwh} "
                f"!= {expected_after}"
            )

        # 6. Battery bounds
        if p.battery_energy_after_kwh < active_min[h] - TOL:
            errors.append(
                f"hour {h}: E_after {p.battery_energy_after_kwh} "
                f"< min {active_min[h]}"
            )
        if p.battery_energy_after_kwh > battery.capacity_kwh + TOL:
            errors.append(f"hour {h}: E_after > capacity")

        # 7. Action match
        if p.battery_action == "idle" and p.battery_kwh > TOL:
            errors.append(f"hour {h}: idle but battery_kwh={p.battery_kwh}")

        prev_energy = p.battery_energy_after_kwh

    # 8. End-of-day neutrality
    if abs(prev_energy - battery.initial_energy_kwh) > TOL:
        errors.append(
            f"end of day: E={prev_energy} != initial={battery.initial_energy_kwh}"
        )

    return errors