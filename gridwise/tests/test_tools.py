# tests/test_tools.py
from app.tools import (
    time_window_to_hours,
    solar_factor,
    reserve_kwh_from_percent,
    validate_hours,
)


def test_time_window_basic():
    assert time_window_to_hours.invoke({"start": "1 PM", "end": "3 PM"}) == {
        "valid": True, "hours": [13, 14], "error": None,
    }


def test_time_window_noon():
    assert time_window_to_hours.invoke({"start": "noon", "end": "2 PM"})["hours"] == [12, 13]


def test_time_window_three_hours():
    assert time_window_to_hours.invoke({"start": "6 PM", "end": "9 PM"})["hours"] == [18, 19, 20]


def test_time_window_24h_format():
    assert time_window_to_hours.invoke({"start": "13:00", "end": "15:00"})["hours"] == [13, 14]


def test_time_window_am_pm_mix():
    assert time_window_to_hours.invoke({"start": "11 AM", "end": "2 PM"})["hours"] == [11, 12, 13]


def test_time_window_midnight():
    assert time_window_to_hours.invoke({"start": "midnight", "end": "5 AM"})["hours"] == [0, 1, 2, 3, 4]


def test_time_window_bad_order():
    r = time_window_to_hours.invoke({"start": "3 PM", "end": "1 PM"})
    assert r["valid"] is False


def test_time_window_non_whole_hour():
    r = time_window_to_hours.invoke({"start": "1:30 PM", "end": "3 PM"})
    assert r["valid"] is False


def test_solar_factor_from_reduction():
    assert solar_factor.invoke({"reduction_percent": 80})["factor"] == 0.2
    assert solar_factor.invoke({"reduction_percent": 75})["factor"] == 0.25
    assert solar_factor.invoke({"reduction_percent": 50})["factor"] == 0.5


def test_solar_factor_from_remaining():
    assert solar_factor.invoke({"remaining_percent": 20})["factor"] == 0.2
    assert solar_factor.invoke({"remaining_percent": 25})["factor"] == 0.25
    assert solar_factor.invoke({"remaining_percent": 50})["factor"] == 0.5


def test_solar_factor_requires_exactly_one_arg():
    assert solar_factor.invoke({})["valid"] is False
    assert solar_factor.invoke({"reduction_percent": 80, "remaining_percent": 20})["valid"] is False


def test_reserve_percent_to_kwh():
    assert reserve_kwh_from_percent.invoke({"percent": 50, "capacity_kwh": 200})["minimum_energy_kwh"] == 100.0
    assert reserve_kwh_from_percent.invoke({"percent": 60, "capacity_kwh": 250})["minimum_energy_kwh"] == 150.0


def test_validate_hours_sorts_and_dedupes():
    assert validate_hours.invoke({"hours": [13, 12, 12]})["hours"] == [12, 13]


def test_validate_hours_rejects_out_of_range():
    assert validate_hours.invoke({"hours": [24]})["valid"] is False
    assert validate_hours.invoke({"hours": [-1]})["valid"] is False


def test_validate_hours_rejects_empty():
    assert validate_hours.invoke({"hours": []})["valid"] is False