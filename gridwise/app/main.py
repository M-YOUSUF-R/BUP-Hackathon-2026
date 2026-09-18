# app/main.py
import os
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import logging
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

from .schemas import (
    OptimizeRequest, OptimizeResponse, HealthResponse, DirectiveInterpretation,
)
from .interpreter import interpret_notes, fallback_all_noop
from .guardrails import validate_interpretation
from .optimizer import solve_plan

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("gridwise")

app = FastAPI(title="GridWise", version="1.0.0")

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

@app.get("/", include_in_schema=False)
def ui_index() -> FileResponse:
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


def _summarize(
    directives: list[DirectiveInterpretation], plan_cost: float
) -> str:
    applied = [d for d in directives if d.applies]
    if not applied:
        return (
            f"No applicable directives. Minimized grid cost "
            f"via LP; total_cost_bdt={plan_cost:.2f}."
        )
    kinds = ", ".join(d.directive_type for d in applied)
    return (
        f"Applied {len(applied)} directive(s) [{kinds}] and minimized "
        f"grid cost via LP; total_cost_bdt={plan_cost:.2f}."
    )


@app.post("/optimize-energy", response_model=OptimizeResponse)
def optimize_energy(req: OptimizeRequest) -> OptimizeResponse:
    # ---- 1. LLM interpretation (with fallback) ----
    try:
        raw = interpret_notes(
            notes=req.operator_notes,
            capacity_kwh=req.battery.capacity_kwh,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM failed, using no_op fallback: %s", exc)
        raw = fallback_all_noop(req.operator_notes)

    # ---- 2. Deterministic guardrails ----
    directives = validate_interpretation(
        raw=raw,
        n_notes=len(req.operator_notes),
        capacity_kwh=req.battery.capacity_kwh,
    )

    # ---- 3. Optimization ----
    try:
        plan, total_grid, total_cost, peak_grid = solve_plan(
            hours=req.hours,
            battery=req.battery,
            directives=directives,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Optimizer failed")
        # Controlled 500 — no stack trace to client.
        raise HTTPException(
            status_code=500,
            detail="Optimization failed; please retry.",
        ) from exc

    # ---- 4. Response ----
    return OptimizeResponse(
        scenario_id=req.scenario_id,
        directive_interpretation=directives,
        hourly_plan=plan,
        total_grid_kwh=total_grid,
        total_cost_bdt=total_cost,
        peak_grid_kwh=peak_grid,
        plan_summary=_summarize(directives, total_cost),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):  # noqa: ANN001
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )