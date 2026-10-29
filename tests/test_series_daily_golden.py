"""Daily results are pinned: every series statistic on the demo store, exactly.

The Shopify app's engine (hosted's /internal/v1/analyze) runs these functions
on daily series, and its output serialises their facts, ``params`` and
``warnings``. Weekly input (0.1.42) must leave every daily result identical:
no changed figure, no new or renamed key, no reworded warning.

``fixtures/series-demo-daily.csv`` is the demo store's daily series, the
synthetic store hosted's demo fixture holds (``analysis-input.demo.v1``; its
March input is the same store cut at 1 March 2026). The golden file was
written by core 0.1.41 (54b0c87) with the locked numpy, scipy, pandas and
statsmodels. Each case calls a function the way hosted does, with its
calibration, plus the options and refusals hosted can reach.

Two comparisons. On the platform the file was recorded on, exact: every float
by its repr, every series by a hash of every value. Elsewhere (CI's Linux),
linear algebra may differ in the last bit between BLAS builds and CPUs, which
is not a change to the code, so floats and each series' length, ends, sum, sum
of squares, minimum and maximum must agree to ``RTOL``. A change to an
estimator moves those by far more. The exact check for hosted is separate: its
analyze output on the demo inputs, diffed byte for byte before and after, on
one machine.

To regenerate (only for a change meant to move daily results, said so in the
PR): ``GOLDEN_WRITE=1 pytest tests/test_series_daily_golden.py -k write``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import os
import platform
from datetime import date
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("statsmodels")

from stats_compass_core.series import (  # noqa: E402
    decompose,
    detect_run,
    forecast,
    its_lift,
    median_gap,
    month_effects,
    repeat_rate,
    verdict,
)

FIXTURES = Path(__file__).parent / "fixtures"
GOLDEN = FIXTURES / "series-daily-golden.json"
RTOL = 1e-9

METHODS = ("mstl", "stl", "fourier")
PERIODS = (7, 365)
# Hosted's calibration (contract config) and promotion settings.
CAL = {"seasonal_band_z": 2.0, "noise_band_z": 2.5, "real_change_alpha": 0.05}
SUMMER40 = (date(2026, 5, 14), date(2026, 6, 3))
PRIOR = (date(2025, 5, 14), date(2025, 6, 3))
OTHER = (date(2025, 11, 24), date(2025, 11, 30))
WEEK = (date(2026, 9, 21), date(2026, 9, 27))
MARCH_WEEK = (date(2026, 2, 23), date(2026, 3, 1))


def _load() -> pd.DataFrame:
    frame = pd.read_csv(FIXTURES / "series-demo-daily.csv", parse_dates=["date"])
    return frame.set_index("date").astype(float)


DATA = _load()


def col(name: str) -> pd.Series:
    return DATA[name].rename(name)


def before(series: pd.Series, day: date) -> pd.Series:
    return series.loc[: pd.Timestamp(day) - pd.Timedelta(days=1)]


def within(series: pd.Series, window: tuple[date, date]) -> pd.Series:
    return series.loc[pd.Timestamp(window[0]) : pd.Timestamp(window[1])]


def returning_rate() -> pd.Series:
    return (DATA["returning_customers"] / DATA["customers_n"]).rename("returning_rate")


# -- canonical form ------------------------------------------------------------


def _num(x: float) -> str:
    """Exact: repr round-trips a float, so equal strings mean equal bits."""
    if math.isnan(x):
        return "nan"
    return repr(float(x))


def _digest(pairs: list[tuple[str, str]]) -> str:
    return hashlib.sha256(json.dumps(pairs).encode()).hexdigest()


def _stats(values: np.ndarray) -> dict:
    """What the tolerant comparison reads: robust to last-bit noise, not to a change."""
    finite = values[np.isfinite(values)]
    return {
        "n_nan": int(np.isnan(values).sum()),
        "sum": _num(finite.sum()) if len(finite) else "nan",
        "sumsq": _num((finite**2).sum()) if len(finite) else "nan",
        "min": _num(finite.min()) if len(finite) else "nan",
        "max": _num(finite.max()) if len(finite) else "nan",
    }


def canonical(obj: Any) -> Any:
    """Everything a result holds, in a form that compares exactly.

    Long series are reduced to their length, ends and a hash of every
    (index, value) pair, so the golden file stays small but any change to any
    value still shows.
    """
    if obj is None or isinstance(obj, (bool, str)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        return _num(obj)
    if isinstance(obj, (date, pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, pd.Series):
        pairs = [(str(k), _num(v) if isinstance(v, (float, np.floating, int, np.integer)) else str(v)) for k, v in obj.items()]
        numeric = pd.api.types.is_numeric_dtype(obj.dtype)
        return {"series": len(pairs), "first": pairs[:1], "last": pairs[-1:], "sha256": _digest(pairs),
                **(_stats(obj.to_numpy(dtype=float)) if numeric else {})}
    if isinstance(obj, pd.DataFrame):
        return {"frame": list(map(str, obj.columns)), **{str(c): canonical(obj[c]) for c in obj.columns}, "attrs": canonical(dict(obj.attrs))}
    if isinstance(obj, np.ndarray):
        numeric = np.issubdtype(obj.dtype, np.number)
        flat = [_num(v) if numeric else str(v) for v in obj.ravel()]
        return {"array": list(obj.shape), "sha256": _digest([(str(i), v) for i, v in enumerate(flat)]),
                **(_stats(obj.astype(float).ravel()) if numeric else {})}
    if dataclasses.is_dataclass(obj):
        return {"type": type(obj).__name__, **{f.name: canonical(getattr(obj, f.name)) for f in dataclasses.fields(obj)}}
    if hasattr(obj, "model_dump"):
        return {"type": type(obj).__name__, **canonical(obj.model_dump())}
    if isinstance(obj, dict):
        return {str(k): canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [canonical(v) for v in obj]
    raise TypeError(f"no canonical form for {type(obj).__name__}")


# -- the cases -----------------------------------------------------------------


def _decomposition_case(dec) -> dict:
    """A decomposition, and the expectations read from it."""
    out = {"dec": canonical(dec)}
    if hasattr(dec, "expected_daily"):
        end = pd.Timestamp(dec.params["end"]).date()
        ahead = pd.date_range(pd.Timestamp(end) + pd.Timedelta(days=1), periods=14)
        out["expected_ahead"] = canonical(dec.expected_daily(ahead[0].date(), ahead[-1].date()))
        out["errors_7"] = canonical(dec.expectation_errors(7))
    return out


def _cases() -> dict[str, Callable[[], Any]]:
    rev, orders, aov = col("net_revenue"), col("order_count"), col("aov")
    rate = returning_rate()
    hist = before(rev, WEEK[0])
    cases: dict[str, Callable[[], Any]] = {}

    for m in METHODS:
        cases[f"decompose/{m}"] = lambda m=m: _decomposition_case(decompose(hist, method=m, periods=PERIODS))
        cases[f"decompose/{m}/exclude"] = lambda m=m: _decomposition_case(
            decompose(hist, method=m, periods=PERIODS, exclude=[(SUMMER40[0], date(2026, 6, 24))])
        )
        cases[f"decompose/{m}/robust"] = lambda m=m: _decomposition_case(
            decompose(hist, method=m, periods=PERIODS, robust=True)
        )
        for ahead in ("flat", "linear"):
            cases[f"decompose/{m}/trend_ahead_{ahead}"] = lambda m=m, ahead=ahead: _decomposition_case(
                decompose(hist, method=m, periods=PERIODS, trend_ahead=ahead)
            )

        def judged(series, kind, aggregate="sum", weights=None, week=WEEK, exclude=(), robust=False, m=m):
            dec = decompose(before(series, week[0]), method=m, periods=PERIODS, exclude=list(exclude))
            if not hasattr(dec, "params"):
                return dec  # insufficient history: hosted reports it as such
            return verdict(
                dec, within(series, week), kind=kind, weights=weights, aggregate=aggregate,
                robust_scale=robust, **CAL,
            )

        cases[f"verdict/{m}/money"] = lambda j=judged: j(rev, "money")
        cases[f"verdict/{m}/money_excluding_promotion"] = lambda j=judged: j(rev, "money", exclude=[(SUMMER40[0], date(2026, 6, 24))])
        cases[f"verdict/{m}/money_robust"] = lambda j=judged: j(rev, "money", robust=True)
        cases[f"verdict/{m}/count"] = lambda j=judged: j(orders, "count")
        cases[f"verdict/{m}/money_mean"] = lambda j=judged: j(aov, "money", "mean", DATA["orders_n"])
        cases[f"verdict/{m}/ratio"] = lambda j=judged: j(rate, "ratio", "mean", DATA["customers_n"])
        cases[f"verdict/{m}/march"] = lambda j=judged: j(rev, "money", week=MARCH_WEEK)

        full = lambda m=m: decompose(rev, method=m, periods=PERIODS)  # noqa: E731
        cases[f"month_effects/{m}"] = lambda full=full: month_effects(full(), level=0.9)
        cases[f"forecast/{m}"] = lambda full=full: forecast(full(), horizon_days=90, level=0.9)
        cases[f"detect_run/{m}"] = lambda full=full: detect_run(full(), min_moves=4, real_change_alpha=0.05)

        def lift(series, kind, m=m, **extra):
            options = dict(
                kind=kind, method=m, periods=PERIODS, prior_window=PRIOR, other_windows=[],
                noise_band="none", post_days=21, baseline_lookback_days=84, level=0.9,
                series_days_before=28, series_days_after=21, **CAL,
            )
            options.update(extra)
            return its_lift(series, SUMMER40, **options)

        cases[f"its_lift/{m}/money"] = lambda lift=lift: lift(rev, "money")
        cases[f"its_lift/{m}/count"] = lambda lift=lift: lift(orders, "count")
        cases[f"its_lift/{m}/other_windows"] = lambda lift=lift: lift(rev, "money", other_windows=[OTHER])
        cases[f"its_lift/{m}/prior_window_band"] = lambda lift=lift: lift(rev, "money", noise_band="prior_window")
        cases[f"its_lift/{m}/no_series"] = lambda lift=lift: lift(rev, "money", series_days_before=None, series_days_after=None)

    # The refusals hosted can reach.
    gap = rev.copy()
    gap.loc["2025-03-01":"2025-03-12"] = np.nan
    sparse_week = within(rev, WEEK).copy()
    sparse_week.iloc[:4] = np.nan
    cases["insufficient/decompose_gap"] = lambda: decompose(gap, method="fourier", periods=PERIODS)
    cases["insufficient/decompose_stl_short"] = lambda: decompose(rev.iloc[:300], method="stl", periods=PERIODS)
    cases["insufficient/decompose_mstl_short"] = lambda: decompose(rev.iloc[:10], method="mstl", periods=PERIODS)
    cases["insufficient/decompose_mstl_annual_dropped"] = lambda: _decomposition_case(
        decompose(rev.iloc[:400], method="mstl", periods=PERIODS)
    )
    cases["insufficient/verdict_too_few_points"] = lambda: verdict(
        decompose(hist, method="fourier", periods=PERIODS), sparse_week, kind="money", **CAL
    )
    cases["insufficient/verdict_history_ends_early"] = lambda: verdict(
        decompose(before(rev, date(2026, 9, 14)), method="fourier", periods=PERIODS),
        within(rev, WEEK), kind="money", **CAL,
    )
    cases["insufficient/verdict_too_short"] = lambda: verdict(
        decompose(rev.iloc[:40], method="fourier", periods=PERIODS), rev.iloc[40:47], kind="money", **CAL
    )
    cases["insufficient/month_effects_no_annual"] = lambda: month_effects(
        decompose(rev.iloc[:400], method="mstl", periods=PERIODS)
    )
    cases["insufficient/forecast_too_short"] = lambda: forecast(
        decompose(rev.iloc[:60], method="fourier", periods=PERIODS), horizon_days=90
    )
    cases["insufficient/its_baseline"] = lambda: its_lift(
        rev, (date(2024, 9, 20), date(2024, 9, 30)), kind="money", method="fourier", periods=PERIODS
    )

    # Customers: seeded cohorts, as hosted passes them.
    rng = np.random.default_rng(20261029)
    counts = (1 + rng.poisson(0.4, 400)).tolist()
    gaps = rng.gamma(2.0, 30.0, 500).round().tolist()
    cases["repeat_rate"] = lambda: repeat_rate(
        counts, level=0.9, min_customers=100, comparison_rate=0.3, comparison_n=1200,
        seasonal_band_z=2.0, real_change_alpha=0.05,
    )
    cases["repeat_rate/no_comparison"] = lambda: repeat_rate(counts, level=0.9)
    cases["repeat_rate/insufficient"] = lambda: repeat_rate(counts[:50], level=0.9)
    cases["median_gap"] = lambda: median_gap(gaps)
    cases["median_gap/insufficient"] = lambda: median_gap(gaps[:5])
    return cases


CASES = _cases()


def _platform() -> str:
    return f"{platform.system()}-{platform.machine()}"


def _run(name: str) -> Any:
    return canonical(CASES[name]())


def _float(s: Any) -> float | None:
    if not isinstance(s, str):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def _close_stats(got: dict, want: dict, path: str) -> list[str]:
    """A series' or array's stats, each against the array's own scale.

    A sum of values that cancel (least-squares residuals sum to about zero)
    is rounding noise of the size of the values, not of the sum: its
    tolerance is RTOL × √n × the array's norm, the bound on rounding in a sum.
    The sum of squares cannot cancel and is compared relatively.
    """
    n = want.get("series", None)
    if n is None:
        n = int(np.prod(want["array"])) if "array" in want else 1
    sumsq = _float(want["sumsq"])
    scale = math.sqrt(max(n, 1) * sumsq) if sumsq is not None and math.isfinite(sumsq) else 0.0
    out = []
    for k in want:
        if k in ("sha256", "first", "last"):
            continue
        if k in ("sum", "min", "max"):
            g, w = _float(got[k]), _float(want[k])
            if g is None or w is None or not (
                (math.isnan(g) and math.isnan(w)) or math.isclose(g, w, rel_tol=RTOL, abs_tol=RTOL * scale)
            ):
                out.append(f"{path}/{k}: {got[k]} != {want[k]}")
        else:
            out += _close(got[k], want[k], f"{path}/{k}")
    return out


def _close(got: Any, want: Any, path: str = "") -> list[str]:
    """Differences beyond RTOL, ignoring exact digests and per-value pairs."""
    if isinstance(want, dict):
        if not isinstance(got, dict) or set(got) != set(want):
            return [f"{path}: keys {sorted(got) if isinstance(got, dict) else got!r} != {sorted(want)}"]
        if "sumsq" in want:
            return _close_stats(got, want, path)
        out = []
        for k in want:
            if k in ("sha256", "first", "last"):
                continue  # exact forms; the stats beside them are compared instead
            out += _close(got[k], want[k], f"{path}/{k}")
        return out
    if isinstance(want, list):
        if not isinstance(got, list) or len(got) != len(want):
            return [f"{path}: {got!r} != {want!r}"]
        return [d for i, (g, w) in enumerate(zip(got, want)) for d in _close(g, w, f"{path}[{i}]")]
    g, w = _float(got), _float(want)
    if g is not None and w is not None:
        if math.isnan(g) and math.isnan(w):
            return []
        if math.isclose(g, w, rel_tol=RTOL, abs_tol=RTOL * 1e-3):
            return []
        return [f"{path}: {got} != {want}"]
    return [] if got == want else [f"{path}: {got!r} != {want!r}"]


def test_every_case_is_pinned():
    if os.environ.get("GOLDEN_WRITE"):
        pytest.skip("writing the golden file")
    golden = json.loads(GOLDEN.read_text())
    assert sorted(golden["cases"]) == sorted(CASES)


@pytest.mark.parametrize("name", sorted(CASES))
def test_daily_results_are_unchanged(name):
    if os.environ.get("GOLDEN_WRITE"):
        pytest.skip("writing the golden file")
    golden = json.loads(GOLDEN.read_text())
    got, want = _run(name), golden["cases"][name]
    if _platform() == golden["recorded_on"]:
        assert got == want
    else:
        assert _close(got, want) == []


def test_write_golden():
    """Writes the golden file when GOLDEN_WRITE is set; otherwise does nothing."""
    if not os.environ.get("GOLDEN_WRITE"):
        return
    cases = {name: _run(name) for name in sorted(CASES)}
    GOLDEN.write_text(json.dumps({"recorded_on": _platform(), "cases": cases}, indent=1, sort_keys=True) + "\n")
