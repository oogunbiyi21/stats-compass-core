"""Weekly input: how often the verdict and the lift raise a false alarm, and how
often they find a real change (0.1.42).

Synthetic weekly sales, labelled by the week ending Sunday: a level of 10,000
a week, an annual swing (none, or ±15%), growth of 10% a year, and noise of 8%
a week (independent, or AR(1) at 0.4, since sell-out weeks follow each other).
For each history length and seed:

- verdict, nothing injected: is the latest week a real change? (false alarm)
- verdict, the latest week +20%: is it found? (power)
- lift, a 4-week window with nothing injected (false alarm), and with +20%
  over its four weeks (power, bias, and whether the 90% interval covers 20%).

Run: python scripts/calibrate_weekly.py [--seeds 200] [--method fourier]
Prints a markdown table. Deterministic: seeds are fixed.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta

import numpy as np
import pandas as pd

from stats_compass_core.series import Insufficient, decompose, its_lift, verdict

LENGTHS = (52, 78, 104, 156)
END = pd.Timestamp("2026-09-27")  # a Sunday
CAL = {"seasonal_band_z": 2.0, "noise_band_z": 2.5, "real_change_alpha": 0.05}
SHIFT = 0.20


def series(weeks: int, seed: int, amplitude: float, ar: float, noise: float = 0.08) -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(end=END, periods=weeks, freq="W-SUN")
    years = (idx - idx[0]).days.to_numpy() / 365.25
    season = 1 + amplitude * np.sin(2 * np.pi * years)
    shocks = rng.standard_normal(weeks)
    e = np.empty(weeks)
    e[0] = shocks[0]
    for i in range(1, weeks):
        e[i] = ar * e[i - 1] + np.sqrt(1 - ar**2) * shocks[i]
    values = 10_000 * (1 + 0.10 * years) * season * (1 + noise * e)
    return pd.Series(values, index=idx)


def judge(s: pd.Series, method: str, weeks: int = 1):
    """The verdict on the last ``weeks`` weeks, from the history before them."""
    dec = decompose(s.iloc[:-weeks], method=method, freq="W")
    if isinstance(dec, Insufficient):
        return dec
    return verdict(dec, s.iloc[-weeks:], kind="money", min_points=1, **CAL)


def lift(s: pd.Series, method: str, window: tuple[date, date]):
    return its_lift(s, window, kind="money", method=method, freq="W", level=0.9, **CAL)


def rate(k: int, n: int) -> str:
    return f"{k / n:.1%}" if n else "n/a"


def cells():
    for weeks in LENGTHS:
        for amplitude in (0.0, 0.15):
            for ar in (0.0, 0.4):
                yield weeks, amplitude, ar


def label(weeks: int, amplitude: float, ar: float) -> str:
    return f"| {weeks} | {'±15%' if amplitude else 'none'} | {'AR(1) 0.4' if ar else 'independent'} "


def verdicts(seeds: int, method: str) -> None:
    print("**The verdict.** False alarm: the share of untouched periods called a real change (the "
          "test's nominal rate is 5%, and the bands make the verdict stricter). Power: the share "
          f"found when every week of the period is +{SHIFT:.0%}.")
    print()
    print("| Weeks of history | Annual swing | Weekly noise | 1 week: false alarm | 1 week: power | 4 weeks: false alarm | 4 weeks: power |")
    print("|---|---|---|---|---|---|---|")
    for weeks, amplitude, ar in cells():
        counts = {k: [0, 0] for k in ("n1", "p1", "n4", "p4")}
        why = set()
        for seed in range(seeds):
            s = series(weeks, 10_000 * weeks + seed, amplitude, ar)
            for span, null_key, hit_key in ((1, "n1", "p1"), (4, "n4", "p4")):
                null = judge(s, method, span)
                if isinstance(null, Insufficient):
                    why.add(f"{span}w {null.reason}, needs {null.needs} {null.unit}")
                    continue
                counts[null_key][0] += null.verdict == "real change"
                counts[null_key][1] += 1
                shifted = s.copy()
                shifted.iloc[-span:] *= 1 + SHIFT
                counts[hit_key][0] += judge(shifted, method, span).verdict == "real change"
                counts[hit_key][1] += 1
        print(label(weeks, amplitude, ar) + "".join(f"| {rate(*counts[k])} " for k in ("n1", "p1", "n4", "p4")) + "|"
              + (f" {'; '.join(sorted(why))}" if why else ""))
    print()


def lifts(seeds: int, method: str) -> None:
    # The promotion: the four weeks ending 10 weeks before the end, so there
    # are weeks after it for the post-window tail.
    last = (END - pd.Timedelta(weeks=10)).date()
    window = (last - timedelta(days=27), last)
    window_weeks = pd.date_range(end=pd.Timestamp(last), periods=4, freq="W-SUN")
    print("**The promotion lift**, a four-week window. False alarm: an untouched window called a real "
          "change. The 90% interval should exclude zero for about 10% of untouched windows. Power: "
          f"the share found when the four weeks are +{SHIFT:.0%}; then the mean estimate and how "
          f"often the 90% interval holds +{SHIFT:.0%}.")
    print()
    print("| Weeks of history | Annual swing | Weekly noise | False alarm | Interval excludes 0 | Power | Mean estimate | Interval holds +20% |")
    print("|---|---|---|---|---|---|---|---|")
    for weeks, amplitude, ar in cells():
        null_real = excl = hit = cover = n = 0
        estimates = []
        why = set()
        for seed in range(seeds):
            s = series(weeks, 10_000 * weeks + seed, amplitude, ar)
            inert = lift(s, method, window)
            if isinstance(inert, Insufficient):
                why.add(f"{inert.reason}, needs {inert.needs} {inert.unit}")
                continue
            n += 1
            null_real += inert.verdict == "real change"
            lo, hi = inert.interval_pct
            excl += lo > 0 or hi < 0
            lifted = s.copy()
            lifted.loc[window_weeks] *= 1 + SHIFT
            found = lift(lifted, method, window)
            hit += found.verdict == "real change"
            estimates.append(found.estimate_pct)
            lo, hi = found.interval_pct
            cover += lo <= 100 * SHIFT <= hi
        mean = f"{np.mean(estimates):+.1f}%" if estimates else "n/a"
        print(label(weeks, amplitude, ar) + f"| {rate(null_real, n)} | {rate(excl, n)} | {rate(hit, n)} | {mean} | {rate(cover, n)} |"
              + (f" {'; '.join(sorted(why))}" if why else ""))
    print()


def run(seeds: int, method: str) -> None:
    print(f"Method `{method}`, {seeds} seeds per row; weekly noise 8%, growth 10% a year.")
    print()
    verdicts(seeds, method)
    lifts(seeds, method)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=200)
    parser.add_argument("--method", default="fourier", choices=["fourier", "stl"])
    args = parser.parse_args()
    run(args.seeds, args.method)
