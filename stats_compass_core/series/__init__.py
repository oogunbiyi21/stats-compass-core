"""Statistics on daily series: series in, statistics out.

Plain functions, not registered tools. Calibration values are arguments with
defaults; anything that draws at random takes a required ``seed``; a shortage
of data is returned as ``Insufficient``, never raised. Results are structured
facts; wording them for a reader is the caller's job.

statsmodels is imported only when a function needs it (install the
``timeseries`` extra).
"""

from stats_compass_core.series._common import Insufficient
from stats_compass_core.series.customers import (
    GapFacts,
    RepeatRateFacts,
    median_gap,
    repeat_rate,
)
from stats_compass_core.series.decompose import (
    Decomposition,
    WeeklyComponents,
    decompose,
)
from stats_compass_core.series.forecast import (
    ForecastFacts,
    ForecastMonth,
    ForecastPoint,
    forecast,
)
from stats_compass_core.series.its import LiftFacts, its_lift
from stats_compass_core.series.runs import RunFacts, detect_run
from stats_compass_core.series.seasonality import (
    MonthEffect,
    MonthEffects,
    month_effects,
)
from stats_compass_core.series.verdict import VerdictFacts, verdict

__all__ = [
    "Decomposition",
    "ForecastFacts",
    "ForecastMonth",
    "ForecastPoint",
    "GapFacts",
    "Insufficient",
    "LiftFacts",
    "MonthEffect",
    "MonthEffects",
    "RepeatRateFacts",
    "RunFacts",
    "VerdictFacts",
    "WeeklyComponents",
    "decompose",
    "detect_run",
    "forecast",
    "its_lift",
    "median_gap",
    "month_effects",
    "repeat_rate",
    "verdict",
]
