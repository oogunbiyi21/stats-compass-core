"""Stores about a year old must not lose what they had the month before.

With 365 days the Fourier fit keeps the annual period, but reading the
seasonal one year back leaves the first year with nothing to measure the
expectation's error against. In 0.1.37 that made headline verdicts
insufficient for 365-418 days of history and the forecast for 365-511, though
both worked below 365 on the weekly pattern alone. Each now falls back to the
weekly pattern when the annual one leaves too little behind it, and says so.
"""

from datetime import date
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from stats_compass_core.series import (  # noqa: E402
    ForecastFacts,
    Insufficient,
    LiftFacts,
    RunFacts,
    VerdictFacts,
    decompose,
    detect_run,
    forecast,
    its_lift,
    month_effects,
    verdict,
)

DEMO = Path(__file__).parent / "fixtures" / "demo_store_daily.csv"


def _demo(column: str = "net_revenue") -> pd.Series:
    return (
        pd.read_csv(DEMO, parse_dates=["date"]).set_index("date")[column].astype(float)
    )


def _codes(warnings) -> list[str]:
    return [w.code for w in warnings]


class TestVerdict:
    @pytest.mark.parametrize("days", [365, 380, 418])
    def test_a_year_old_store_gets_a_verdict_on_the_weekly_pattern(self, days):
        y = _demo()
        dec = decompose(y.iloc[-(days + 7) : -7], method="fourier")
        assert dec.params["periods_used"] == [7, 365]
        facts = verdict(dec, y.iloc[-7:], kind="money")
        assert isinstance(facts, VerdictFacts)
        assert facts.params["annual_fallback"] is True
        assert "ANNUAL_DROPPED" in _codes(facts.warnings)

    def test_with_enough_history_the_annual_pattern_stays(self):
        y = _demo()
        dec = decompose(y.iloc[-(500 + 7) : -7], method="fourier")
        facts = verdict(dec, y.iloc[-7:], kind="money")
        assert facts.params["annual_fallback"] is False


class TestForecast:
    @pytest.mark.parametrize("days", [365, 450, 511])
    def test_a_year_old_store_gets_a_forecast(self, days):
        facts = forecast(decompose(_demo().iloc[-days:], method="fourier"))
        assert isinstance(facts, ForecastFacts)
        assert facts.params["annual_fallback"] is True
        assert "ANNUAL_DROPPED" in _codes(facts.warnings)

    def test_with_enough_history_the_annual_pattern_stays(self):
        facts = forecast(decompose(_demo().iloc[-600:], method="fourier"))
        assert facts.params["annual_fallback"] is False


class TestRunsAndLifts:
    def test_runs_on_a_year_old_store(self):
        result = detect_run(
            decompose(_demo("new_customers").iloc[-420:], method="fourier")
        )
        assert not isinstance(result, Insufficient)
        assert isinstance(result, RunFacts) and result.params["annual_fallback"] is True

    def test_a_lift_soon_after_a_year_of_history(self):
        """A promotion in month 13, analysed five weeks later: the annual
        pattern is kept (over 365 days) but leaves the reference too few
        positions once the window and its tail are excluded."""
        y = _demo().loc["2025-04-15":"2026-06-30"]
        facts = its_lift(
            y, (date(2026, 5, 14), date(2026, 5, 24)), kind="money", method="fourier"
        )
        assert isinstance(facts, LiftFacts)
        assert facts.params["annual_fallback"] is True


class TestSeasonalityStillNeedsTheYear:
    def test_month_effects_do_not_fall_back(self):
        assert not isinstance(
            month_effects(decompose(_demo().iloc[-365:], method="fourier")),
            Insufficient,
        )
        result = month_effects(decompose(_demo().iloc[-364:], method="fourier"))
        assert isinstance(result, Insufficient) and result.reason == "NO_ANNUAL"
