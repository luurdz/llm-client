import pytest

from agent.tools import make_tools


def get_booking_tool():
    """New get_booking tool with its own attempt counter."""
    return make_tools(None)[0]


def test_tool_has_name_and_description():
    booking_tool = get_booking_tool()

    assert booking_tool.name == "get_booking"
    assert "localizador" in booking_tool.description.lower()


@pytest.mark.parametrize("locator", ["RSV7K", "RSV7K22", "RSVIK2", "RSVOK2", "RSV-K2", ""])
def test_invalid_format_returns_rs1001(locator: str):
    result = get_booking_tool().invoke({"locator": locator})

    assert result.startswith("ERROR RS-1001")
    assert "formato" in result


def test_unknown_locator_returns_rs1001():
    result = get_booking_tool().invoke({"locator": "ABC123"})

    assert result.startswith("ERROR RS-1001")
    assert "no existe" in result


def test_healthy_booking_returns_data_without_error():
    result = get_booking_tool().invoke({"locator": "RSV5M4"})

    assert "Sofía Ruiz" in result
    assert "confirmada" in result
    assert "ERROR" not in result
    assert "None" not in result


def test_locator_is_normalized():
    result = get_booking_tool().invoke({"locator": "  rsv5m4 "})

    assert "Sofía Ruiz" in result


def test_blocked_booking_reports_rs2210_once():
    result = get_booking_tool().invoke({"locator": "RSV7K2"})

    assert "bloqueada" in result
    assert result.count("RS-2210") == 1


def test_transient_error_fails_first_call_and_succeeds_on_retry():
    booking_tool = get_booking_tool()

    first = booking_tool.invoke({"locator": "RSV3P8"})
    second = booking_tool.invoke({"locator": "RSV3P8"})

    assert "RS-3100" in first
    assert "Martín Gómez" in second
    assert "RS-3100" not in second


def test_transient_error_counter_is_per_tool_instance():
    get_booking_tool().invoke({"locator": "RSV3P8"})

    result = get_booking_tool().invoke({"locator": "RSV3P8"})

    assert "RS-3100" in result
