# app/schemas.py
from typing import Literal, Optional
from pydantic import BaseModel, Field, field_validator

DirectiveType = Literal[
    "solar_reduction",
    "minimum_battery_reserve",
    "no_charge_window",
    "no_discharge_window",
    "max_grid_window",
    "no_op",
]

BatteryAction = Literal["charge", "discharge", "idle"]


# ---------- Request ----------

class HourInput(BaseModel):
    hour: int = Field(ge=0, le=23)
    demand_kwh: float = Field(ge=0)
    solar_kwh: float = Field(ge=0)
    tariff_bdt_per_kwh: float = Field(ge=0)


class BatteryInput(BaseModel):
    capacity_kwh: float = Field(gt=0)
    initial_energy_kwh: float = Field(ge=0)
    minimum_energy_kwh: float = Field(ge=0)
    max_charge_kwh_per_hour: float = Field(ge=0)
    max_discharge_kwh_per_hour: float = Field(ge=0)


class OptimizeRequest(BaseModel):
    scenario_id: str
    operator_notes: list[str] = Field(min_length=1, max_length=3)
    hours: list[HourInput] = Field(min_length=24, max_length=24)
    battery: BatteryInput

    @field_validator("operator_notes")
    @classmethod
    def no_empty_notes(cls, v: list[str]) -> list[str]:
        if any(not n.strip() for n in v):
            raise ValueError("operator_notes must be non-empty strings")
        return v

    @field_validator("hours")
    @classmethod
    def hours_are_0_to_23(cls, v: list[HourInput]) -> list[HourInput]:
        if [h.hour for h in v] != list(range(24)):
            raise ValueError("hours must be exactly 0..23 in order")
        return v


# ---------- LLM structured output ----------

class RawDirective(BaseModel):
    """What the LLM must produce for one operator note."""
    note_index: int = Field(description="0-based index of the note")
    applies: bool = Field(description="true for real directives, false for no_op")
    directive_type: DirectiveType
    hours: Optional[list[int]] = Field(
        default=None,
        description="Unique ints 0..23, ascending. Start inclusive, end exclusive.",
    )
    factor: Optional[float] = Field(
        default=None,
        description="solar_reduction only: usable fraction REMAINING in [0,1].",
    )
    minimum_energy_kwh: Optional[float] = Field(
        default=None, description="minimum_battery_reserve only: absolute kWh."
    )
    max_grid_kwh: Optional[float] = Field(
        default=None, description="max_grid_window only: kWh cap."
    )
    explanation: str = Field(description="Short rationale.")


class RawInterpretation(BaseModel):
    directives: list[RawDirective]


# ---------- Response ----------

class DirectiveInterpretation(BaseModel):
    note_index: int
    applies: bool
    directive_type: DirectiveType
    structured_adjustment: Optional[dict]  # null for no_op
    explanation: str


class HourlyPlanEntry(BaseModel):
    hour: int
    grid_kwh: float
    solar_used_kwh: float
    battery_action: BatteryAction
    battery_kwh: float
    battery_energy_after_kwh: float


class OptimizeResponse(BaseModel):
    scenario_id: str
    directive_interpretation: list[DirectiveInterpretation]
    hourly_plan: list[HourlyPlanEntry]
    total_grid_kwh: float
    total_cost_bdt: float
    peak_grid_kwh: float
    plan_summary: str

class LLMHealthInfo(BaseModel):
    ok: bool
    provider: str
    model: str
    latency_ms: Optional[int] = None
    error_type: Optional[str] = None
    message: str = ""
    cached: bool = False


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    llm: Optional[LLMHealthInfo] = None
