"""Statistics on daily series: series in, statistics out.

Plain functions, not registered tools. Calibration values are arguments with
defaults; anything that draws at random takes a required ``seed``; a shortage
of data is returned as ``Insufficient``, never raised. Results are structured
facts; wording them for a reader is the caller's job.

statsmodels is imported only when a function needs it (install the
``timeseries`` extra).
"""

from stats_compass_core.series._common import Insufficient
from stats_compass_core.series.decompose import (
    Decomposition,
    WeeklyComponents,
    decompose,
)
from stats_compass_core.series.seasonality import (
    MonthEffect,
    MonthEffects,
    month_effects,
)

__all__ = [
    "Decomposition",
    "Insufficient",
    "MonthEffect",
    "MonthEffects",
    "WeeklyComponents",
    "decompose",
    "month_effects",
]
