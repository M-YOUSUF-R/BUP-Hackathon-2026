# app/optimizer.py
import pulp
from .schemas import (
    HourInput, BatteryInput, DirectiveInterpretation, HourlyPlanEntry,
)


def _apply_directives(
    hours: list[HourInput],
    battery: BatteryInput,
    directives: list[DirectiveInterpretation],
):
    """Translate directives into concrete per-hour constraint values."""
    eff_solar = [h.solar_kwh for h in hours]
    active_min = [battery.minimum_energy_kwh] * 24
    no_charge_h: set[int] = set()
    no_discharge_h: set[int] = set()
    grid_cap: list[float | None] = [None] * 24

    for d in directives:
        if not d.applies or d.structured_adjustment is None:
            continue
        adj = d.structured_adjustment
        hrs = adj["hours"]

        if d.directive_type == "solar_reduction":
            f = float(adj["factor"])
            for h in hrs:
                eff_solar[h] *= f

        elif d.directive_type == "minimum_battery_reserve":
            floor = float(adj["minimum_energy_kwh"])
            for h in hrs:
                active_min[h] = max(active_min[h], floor)

        elif d.directive_type == "no_charge_window":
            no_charge_h.update(hrs)

        elif d.directive_type == "no_discharge_window":
            no_discharge_h.update(hrs)

        elif d.directive_type == "max_grid_window":
            cap = float(adj["max_grid_kwh"])
            for h in hrs:
                grid_cap[h] = cap

    return eff_solar, active_min, no_charge_h, no_discharge_h, grid_cap


def solve_plan(
    hours: list[HourInput],
    battery: BatteryInput,
    directives: list[DirectiveInterpretation],
) -> tuple[list[HourlyPlanEntry], float, float, float]:
    """Build and solve the LP. Returns (plan, total_grid, total_cost, peak_grid)."""
    eff_solar, active_min, no_charge_h, no_discharge_h, grid_cap = (
        _apply_directives(hours, battery, directives)
    )

    demand = [h.demand_kwh for h in hours]
    tariff = [h.tariff_bdt_per_kwh for h in hours]

    cap = battery.capacity_kwh
    init = battery.initial_energy_kwh
    max_ch = battery.max_charge_kwh_per_hour
    max_dis = battery.max_discharge_kwh_per_hour

    prob = pulp.LpProblem("gridwise", pulp.LpMinimize)

    grid = [pulp.LpVariable(f"grid_{h}", lowBound=0) for h in range(24)]
    solar = [pulp.LpVariable(f"solar_{h}", lowBound=0) for h in range(24)]
    chg = [pulp.LpVariable(f"chg_{h}", lowBound=0) for h in range(24)]
    dis = [pulp.LpVariable(f"dis_{h}", lowBound=0) for h in range(24)]
    E = [pulp.LpVariable(f"E_{h}", lowBound=0, upBound=cap) for h in range(24)]

    # Objective: minimize grid cost.
    prob += pulp.lpSum(grid[h] * tariff[h] for h in range(24))

    for h in range(24):
        prev = init if h == 0 else E[h - 1]

        # Energy balance: grid + solar_used + discharge = demand + charge
        prob += grid[h] + solar[h] + dis[h] == demand[h] + chg[h]

        # Solar cap (effective after directives).
        prob += solar[h] <= eff_solar[h]

        # Rate limits.
        prob += chg[h] <= max_ch
        prob += dis[h] <= max_dis

        # Battery state transition and lower bound.
        prob += E[h] == prev + chg[h] - dis[h]
        prob += E[h] >= active_min[h]

        # Directive bans / caps.
        if h in no_charge_h:
            prob += chg[h] == 0
        if h in no_discharge_h:
            prob += dis[h] == 0
        if grid_cap[h] is not None:
            prob += grid[h] <= grid_cap[h]

    # End-of-day neutrality.
    prob += E[23] == init

    status = prob.solve(pulp.PULP_CBC_CMD(msg=0))
    if pulp.LpStatus[status] != "Optimal":
        raise RuntimeError(f"LP failed: {pulp.LpStatus[status]}")

    plan: list[HourlyPlanEntry] = []
    for h in range(24):
        c = chg[h].value() or 0.0
        d = dis[h].value() or 0.0
        g = grid[h].value() or 0.0
        s = solar[h].value() or 0.0
        e = E[h].value() or 0.0

        # Tolerate tiny solver noise.
        if c > 1e-6 and c >= d:
            action, kwh = "charge", c
        elif d > 1e-6:
            action, kwh = "discharge", d
        else:
            action, kwh = "idle", 0.0

        plan.append(
            HourlyPlanEntry(
                hour=h,
                grid_kwh=round(max(g, 0.0), 6),
                solar_used_kwh=round(max(s, 0.0), 6),
                battery_action=action,
                battery_kwh=round(max(kwh, 0.0), 6),
                battery_energy_after_kwh=round(max(e, 0.0), 6),
            )
        )

    total_grid = sum(p.grid_kwh for p in plan)
    total_cost = sum(p.grid_kwh * tariff[p.hour] for p in plan)
    peak_grid = max(p.grid_kwh for p in plan)

    return plan, round(total_grid, 6), round(total_cost, 6), round(peak_grid, 6)
