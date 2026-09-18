# app/interpreter.py
"""
Interpreter layer. Two paths:

1. Tool-calling agent (primary) — forces the LLM to use deterministic math.
2. Plain structured output (fallback) — if the agent loop fails for any
   reason, we still get *something* interpretable.

The caller receives a RawInterpretation either way. Guardrails are applied
downstream, unchanged.
"""
from __future__ import annotations

import logging
import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from .schemas import RawDirective, RawInterpretation
from .tool_agent import interpret_with_tools

logger = logging.getLogger(__name__)


SIMPLE_PROMPT = """You convert campus operator notes into structured energy \
directives. This is a fallback path with NO tools — be conservative.

Return one directive per note, in note_index order.
- solar_reduction: factor is the USABLE FRACTION REMAINING (80% reduction -> 0.2)
- minimum_battery_reserve: absolute kWh (for "% of capacity", multiply by capacity)
- no_charge_window / no_discharge_window / max_grid_window
- no_op: unrelated note; applies=false; numeric fields null

Time windows use START inclusive, END exclusive: 1 PM to 3 PM -> [13,14].
Hours must be unique ints 0..23, ascending. Never invent directive types.
"""


def _simple_structured_fallback(
    notes: list[str], capacity_kwh: float
) -> RawInterpretation:
    llm = ChatOpenAI(
        model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        temperature=0,
        timeout=float(os.getenv("LLM_TIMEOUT_SECONDS", "15")),
        max_retries=0,
    ).with_structured_output(RawInterpretation)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SIMPLE_PROMPT),
            ("human", "Battery capacity_kwh = {cap}\n\nNotes:\n{notes}"),
        ]
    )
    return (prompt | llm).invoke(
        {
            "cap": capacity_kwh,
            "notes": "\n".join(f"{i}. {n}" for i, n in enumerate(notes)),
        }
    )


def fallback_all_noop(notes: list[str]) -> RawInterpretation:
    return RawInterpretation(
        directives=[
            RawDirective(
                note_index=i,
                applies=False,
                directive_type="no_op",
                hours=None,
                factor=None,
                minimum_energy_kwh=None,
                max_grid_kwh=None,
                explanation="LLM unavailable; note treated as no_op.",
            )
            for i in range(len(notes))
        ]
    )


def interpret_notes(notes: list[str], capacity_kwh: float) -> RawInterpretation:
    """Primary entry point used by the API.

    Order of attempts:
      1. Tool-calling agent
      2. Structured output fallback
      3. All-no_op fallback (never raises)
    """
    # 1. agent
    try:
        result = interpret_with_tools(notes, capacity_kwh)
        if result and result.directives:
            return result
        logger.warning("agent returned empty directives; falling back")
    except Exception as exc:  # noqa: BLE001
        logger.warning("tool agent failed: %s", exc)

    # 2. structured output
    try:
        result = _simple_structured_fallback(notes, capacity_kwh)
        if result and result.directives:
            return result
    except Exception as exc:  # noqa: BLE001
        logger.warning("structured fallback failed: %s", exc)

    # 3. all no_op
    return fallback_all_noop(notes)
