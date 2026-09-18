# app/tool_agent.py
"""
Tool-calling agent that forces the LLM to delegate all arithmetic to tools.

The agent's toolset is:
  - time_window_to_hours
  - solar_factor
  - reserve_kwh_from_percent
  - validate_hours
  - emit_interpretation   <-- the LLM must call this last with the final schema

The loop terminates on the first emit_interpretation call. If the model tries
to finish without calling it, we remind it and continue.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI

from .schemas import RawDirective, RawInterpretation
from .tools import MATH_TOOLS, MATH_TOOLS_BY_NAME

logger = logging.getLogger(__name__)

MAX_ITERATIONS = 12  # safety bound


# ---- final-output tool ----
# LangChain generates the JSON schema from the Pydantic annotation.
@tool
def emit_interpretation(directives: list[RawDirective]) -> dict[str, Any]:
    """Emit the FINAL interpretation for all operator notes.

    Call this exactly once, LAST, after all math has been done with the
    other tools. Do not include this call's output in any subsequent step.
    """
    # The body never runs; we intercept the call in the loop below.
    return {"emitted": True}


AGENT_TOOLS = MATH_TOOLS + [emit_interpretation]
AGENT_TOOLS_BY_NAME = {t.name: t for t in AGENT_TOOLS}


SYSTEM_PROMPT = """You are a structured interpreter that converts campus
operator notes into energy directives for a 24-hour scheduling system.

You have access to deterministic tools. You MUST use them for ALL arithmetic.
Do NOT compute hours, factors, or kWh yourself. Always call the tool.

AVAILABLE TOOLS:
- time_window_to_hours(start, end)
    Convert a whole-hour window into hour indices. Start inclusive, end exclusive.
    Examples: ("1 PM","3 PM") -> [13,14]; ("noon","2 PM") -> [12,13];
              ("6 PM","9 PM") -> [18,19,20].
- solar_factor(reduction_percent=None, remaining_percent=None)
    Convert a solar reduction or "fraction remaining" into the
    solar_reduction.factor (usable fraction in [0,1]).
    Pass EITHER reduction_percent=80 for "80% reduction"
    OR       remaining_percent=20 for "20% of forecast".
- reserve_kwh_from_percent(percent, capacity_kwh)
    Convert "X% of battery capacity" into absolute kWh.
- validate_hours(hours)
    Sort, dedupe, and range-check a list of hours. Use before final emit.
- emit_interpretation(directives)
    Call this LAST, exactly once, with the final interpretation.

SUPPORTED DIRECTIVES:
- solar_reduction:          hours + factor
- minimum_battery_reserve:  hours + minimum_energy_kwh
- no_charge_window:         hours
- no_discharge_window:      hours
- max_grid_window:          hours + max_grid_kwh
- no_op:                    no fields; applies=false

RULES:
- Exactly one directive per note, in note_index order 0..N-1.
- Use no_op for unrelated notes: applies=false, all numeric fields null.
- For every other directive: applies=true.
- Hours must be unique ints 0..23 in ascending order.
- Do NOT invent directive types.
- Do NOT guess numbers. If a note lacks a number you need, use no_op.

WORKFLOW for each note:
1. Decide the directive type (or no_op).
2. If it has a time window, call time_window_to_hours.
3. If it has a percentage, call solar_factor or reserve_kwh_from_percent.
4. If it has hours, call validate_hours.
5. After all notes are processed, call emit_interpretation ONCE.
"""


def _build_llm():
    import os
    return ChatOpenAI(
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "20")),
        max_retries=0,
    ).bind_tools(AGENT_TOOLS)


def _json(obj: Any) -> str:
    try:
        return json.dumps(obj, default=str)
    except Exception:
        return json.dumps({"valid": False, "error": "unserializable tool result"})


def _execute_tool_call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Run a math tool safely. Never raises."""
    if name == "emit_interpretation":
        # handled in loop; this is a fallback if somehow reached
        return {"valid": False, "error": "emit called outside loop"}

    tool_obj = MATH_TOOLS_BY_NAME.get(name)
    if tool_obj is None:
        return {"valid": False, "error": f"unknown tool: {name}"}

    try:
        result = tool_obj.invoke(args)
        return result if isinstance(result, dict) else {"valid": True, "result": result}
    except Exception as exc:  # noqa: BLE001
        logger.warning("tool %s raised: %s", name, exc)
        return {"valid": False, "error": f"{type(exc).__name__}: {exc}"}


def interpret_with_tools(
    notes: list[str],
    capacity_kwh: float,
) -> RawInterpretation:
    """Run the tool-calling loop. Returns the final RawInterpretation.

    Raises RuntimeError if the model does not emit within MAX_ITERATIONS.
    """
    llm = _build_llm()

    notes_block = "\n".join(f"{i}. {n}" for i, n in enumerate(notes))
    messages: list[Any] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Battery capacity_kwh = {capacity_kwh}\n\n"
                f"Operator notes (index 0..{len(notes) - 1}):\n{notes_block}\n\n"
                "Interpret every note using the tools, then call "
                "emit_interpretation exactly once with the final directives."
            )
        ),
    ]

    for iteration in range(MAX_ITERATIONS):
        ai: AIMessage = llm.invoke(messages)
        messages.append(ai)

        tool_calls = getattr(ai, "tool_calls", None) or []
        if not tool_calls:
            # Model tried to answer without emitting. Nudge and retry.
            messages.append(
                HumanMessage(
                    content=(
                        "You must call emit_interpretation with the final "
                        "directives before finishing. Do that now."
                    )
                )
            )
            continue

        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("args") or {}
            call_id = tc.get("id") or f"call_{iteration}"

            if name == "emit_interpretation":
                # Parse and return.
                try:
                    return RawInterpretation(
                        directives=[RawDirective(**d) for d in args["directives"]]
                    )
                except Exception as exc:  # noqa: BLE001
                    # Let the model fix its own output.
                    messages.append(
                        ToolMessage(
                            content=_json(
                                {
                                    "valid": False,
                                    "error": f"schema error: {exc}",
                                }
                            ),
                            tool_call_id=call_id,
                        )
                    )
                    continue

            result = _execute_tool_call(name, args)
            messages.append(
                ToolMessage(content=_json(result), tool_call_id=call_id)
            )

    raise RuntimeError(
        f"agent failed to emit interpretation within {MAX_ITERATIONS} iterations"
    )