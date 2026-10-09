"""A workflow step's summary is text, never a format string (re-scan F6, 9 Oct 2026).

run_step called summary_template.format(result=...) on a string that callers had
already built with column names or the date column in it, so braces in a column
header became format fields. A width allocated as much memory as it asked for
(dev-95 measured 100 MB in 0.19 s from one histogram step on production), and a
field could walk attributes from the step's result. Column names arrive through
data (CSV and Sheet headers, rename_columns, date_column), so no server-side
guard can stop them. No template used {result}; the summary is now used as given.
"""

import pandas as pd
import pytest

from stats_compass_core.state import DataFrameState
from stats_compass_core.workflows.configs import EDAConfig
from stats_compass_core.workflows.eda_report import RunEDAReportInput, run_eda_report

# Small widths: against the old code these allocate a few kilobytes, not gigabytes.
BRACED = [
    "{result.__class__.__name__:>5000}",  # attribute walk to a plain string, with a width
    "{result.__class__}",  # attribute walk alone
    "{result:>20}",  # the plain width the report shows
    "{0}",  # positional field
    "price {unclosed",  # a lone brace
]


def _histogram_summary(column):
    state = DataFrameState()
    state.set_dataframe(pd.DataFrame({column: [1.0, 2.0, 3.0, 4.0, 5.0]}), "t", "test")
    result = run_eda_report(state, RunEDAReportInput(
        dataframe_name="t",
        config=EDAConfig(
            include_describe=False, include_correlations=False, include_missing_analysis=False,
            include_quality_report=False, generate_histograms=True, generate_bar_charts=False,
        ),
    ))
    step = next(s for s in result.steps if s.step_name == f"histogram_{column}")
    return step


@pytest.mark.parametrize("column", BRACED)
def test_a_braced_column_name_is_shown_as_written(column):
    step = _histogram_summary(column)
    assert step.status == "success"
    assert step.summary == f"Generated histogram for '{column}'"


def test_run_step_never_formats_its_template():
    from stats_compass_core.workflows.utils import run_step

    step = run_step("s", 0, lambda state, params: {"ok": True}, None, None, "{result!r:>99999}")
    assert step.summary == "{result!r:>99999}"
