"""T6.9: each tool on the Stage A path that could return a confident wrong number.

One test per hazard found in the sweep (docs/audit/unattended.md). Each sets up
the input that produced a plausible-looking wrong answer and asserts the tool now
refuses it or says so in a named warning.
"""

import numpy as np
import pandas as pd
import pytest

from stats_compass_core.cleaning.clean_dates import CleanDatesInput, clean_dates
from stats_compass_core.eda.correlations import CorrelationsInput, correlations
from stats_compass_core.eda.data_quality import (
    AnalyzeMissingDataInput,
    DataQualityReportInput,
    DetectOutliersInput,
    analyze_missing_data,
    data_quality_report,
    detect_outliers,
)
from stats_compass_core.eda.describe import DescribeInput, describe
from stats_compass_core.eda.hypothesis_tests import (
    TTestInput,
    ZTestInput,
    t_test,
    z_test,
)
from stats_compass_core.ml.timeseries.arima import (
    FindOptimalARIMAInput,
    FitARIMAInput,
    ForecastARIMAInput,
    InferFrequencyInput,
    StationarityTestInput,
    check_stationarity,
    find_optimal_arima,
    fit_arima,
    forecast_arima,
    infer_frequency,
)
from stats_compass_core.results import OperationError
from stats_compass_core.state import DataFrameState
from stats_compass_core.transforms.groupby_aggregate import (
    ColumnAggregation,
    GroupByAggregateInput,
    groupby_aggregate,
)
from stats_compass_core.transforms.pivot import PivotInput, pivot
from stats_compass_core.transforms.split_column_by_group import (
    SplitColumnByGroupInput,
    split_column_by_group,
)
from stats_compass_core.workflows import run_timeseries_forecast
from stats_compass_core.workflows.configs import TimeSeriesConfig
from stats_compass_core.workflows.timeseries import RunTimeseriesForecastInput


def _state(frame: pd.DataFrame, name: str = "d") -> DataFrameState:
    state = DataFrameState()
    state.set_dataframe(frame, name, operation="test")
    return state


def _codes(result) -> list[str]:
    return [w.code for w in result.warnings]


def _logged(state: DataFrameState) -> list[str]:
    return [h.details["code"] for h in state.get_history() if h.operation == "warning"]


def _daily(n: int = 60, start: str = "2024-01-01") -> pd.DataFrame:
    rng = np.random.default_rng(7)
    values = np.zeros(n)
    values[0] = 100
    for i in range(1, n):
        values[i] = 0.7 * values[i - 1] + 30 + rng.normal(0, 5)
    return pd.DataFrame(
        {"date": pd.date_range(start, periods=n, freq="D"), "revenue": values}
    )


# =============================================================================
# Cleaning
# =============================================================================


class TestCleanDates:
    def test_unparseable_dates_are_counted_not_silently_coerced(self):
        frame = pd.DataFrame(
            {"date": ["2024-01-01", "garbage", "2024-01-03"], "orders": [1, 2, 3]}
        )
        state = _state(frame)
        result = clean_dates(
            state, CleanDatesInput(dataframe_name="d", date_column="date")
        )
        assert "UNPARSEABLE_VALUES" in _codes(result)
        assert "UNPARSEABLE_VALUES" in _logged(state)

    def test_filled_dates_are_flagged_as_invented(self):
        """ffill gives an order the previous row's date: a date nobody recorded."""
        frame = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", None, "2024-01-03"]),
                "orders": [1, 2, 3],
            }
        )
        result = clean_dates(
            _state(frame), CleanDatesInput(dataframe_name="d", date_column="date")
        )
        assert "DATES_FILLED" in _codes(result)

    def test_requested_gap_filling_that_cannot_run_says_so(self):
        """pd.infer_freq returns None exactly when there are gaps to fill."""
        frame = _daily(10).drop(index=[3, 4]).reset_index(drop=True)
        result = clean_dates(
            _state(frame),
            CleanDatesInput(
                dataframe_name="d", date_column="date", create_missing_dates=True
            ),
        )
        assert result.rows_after == 8
        assert "GAPS_NOT_FILLED" in _codes(result)


# =============================================================================
# Transforms
# =============================================================================


class TestGroupbyAggregate:
    def test_rows_with_a_missing_key_are_reported(self):
        """Revenue by discount code silently lost every order without a code."""
        frame = pd.DataFrame(
            {
                "discount_code": [None, "SPRING", "SPRING", None, None],
                "revenue": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        state = _state(frame)
        result = groupby_aggregate(
            state,
            GroupByAggregateInput(
                dataframe_name="d",
                by=["discount_code"],
                aggregations=[ColumnAggregation(column="revenue", functions=["sum"])],
            ),
        )
        warning = next(w for w in result.warnings if w.code == "NULL_GROUP_KEYS")
        assert warning.columns == ["discount_code"]
        assert "3 of 5" in warning.message

    def test_a_group_with_no_values_does_not_quietly_sum_to_zero(self):
        """pandas sums an all-null group to 0: 'no data' reported as 'nothing sold'."""
        frame = pd.DataFrame(
            {
                "week": ["w1", "w1", "w2", "w2"],
                "net_revenue": [10.0, -4.0, np.nan, np.nan],
            }
        )
        result = groupby_aggregate(
            _state(frame),
            GroupByAggregateInput(
                dataframe_name="d",
                by=["week"],
                aggregations=[
                    ColumnAggregation(column="net_revenue", functions=["sum"])
                ],
            ),
        )
        warning = next(w for w in result.warnings if w.code == "NULL_GROUP_SUMMED")
        assert warning.columns == ["net_revenue"]
        assert "w2" in warning.message

    def test_summing_numbers_stored_as_text_is_flagged(self):
        frame = pd.DataFrame({"store": ["a", "a"], "revenue": ["10", "20"]})
        result = groupby_aggregate(
            _state(frame),
            GroupByAggregateInput(
                dataframe_name="d",
                by=["store"],
                aggregations=[ColumnAggregation(column="revenue", functions=["sum"])],
            ),
        )
        assert "NUMERIC_AS_TEXT" in _codes(result)


class TestPivot:
    def test_cells_built_from_several_rows_are_flagged(self):
        """Order-level rows pivoted by day x channel average to order value."""
        frame = pd.DataFrame(
            {
                "day": ["mon", "mon", "mon", "tue"],
                "channel": ["web", "web", "pos", "web"],
                "revenue": [10.0, 30.0, 5.0, 7.0],
            }
        )
        result = pivot(
            _state(frame),
            PivotInput(
                dataframe_name="d", index="day", columns="channel", values="revenue"
            ),
        )
        warning = next(w for w in result.warnings if w.code == "CELLS_AGGREGATED")
        assert "mean" in warning.message

    def test_one_row_per_cell_is_quiet(self):
        frame = pd.DataFrame(
            {
                "day": ["mon", "mon", "tue"],
                "channel": ["web", "pos", "web"],
                "revenue": [10.0, 5.0, 7.0],
            }
        )
        result = pivot(
            _state(frame),
            PivotInput(
                dataframe_name="d", index="day", columns="channel", values="revenue"
            ),
        )
        assert result.warnings == []


class TestSplitColumnByGroup:
    def test_numeric_group_values_are_matched(self):
        frame = pd.DataFrame(
            {"cohort": [1, 1, 2, 2], "spend": [10.0, 11.0, 20.0, 21.0]}
        )
        state = _state(frame)
        split_column_by_group(
            state,
            SplitColumnByGroupInput(
                dataframe_name="d",
                value_column="spend",
                group_column="cohort",
                save_as="w",
            ),
        )
        wide = state.get_dataframe("w")
        assert wide["1"].tolist() == [10.0, 11.0]
        assert wide["2"].tolist() == [20.0, 21.0]


# =============================================================================
# EDA
# =============================================================================


def _null_text_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "email": ["a@x.com", "null", "nan", "", "b@x.com", "N/A"],
            "spend": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }
    )


class TestMissingData:
    def test_missing_values_spelled_as_text_are_reported(self):
        """Six rows, four 'missing', and the report used to say all was well."""
        state = _state(_null_text_frame())
        result = analyze_missing_data(
            state, AnalyzeMissingDataInput(dataframe_name="d")
        )
        warning = next(w for w in result.warnings if w.code == "MISSING_AS_TEXT")
        assert warning.columns == ["email"]

    def test_quality_report_too(self):
        result = data_quality_report(
            _state(_null_text_frame()), DataQualityReportInput(dataframe_name="d")
        )
        assert "MISSING_AS_TEXT" in _codes(result)


class TestDescribe:
    def test_numbers_stored_as_text_are_not_left_out_quietly(self):
        frame = pd.DataFrame({"revenue": ["10", "20", "nan"], "units": [1, 2, 3]})
        result = describe(_state(frame), DescribeInput(dataframe_name="d"))
        assert "revenue" not in result.columns_analyzed
        warning = next(w for w in result.warnings if w.code == "NUMERIC_AS_TEXT")
        assert warning.columns == ["revenue"]


class TestCorrelations:
    def test_correlation_from_a_handful_of_rows_is_flagged(self):
        n = 40
        rng = np.random.default_rng(8)
        a = rng.normal(size=n)
        b = np.full(n, np.nan)
        b[:3] = [1.0, 2.0, 3.0]  # overlaps with a on three rows only
        result = correlations(
            _state(pd.DataFrame({"a": a, "b": b})),
            CorrelationsInput(dataframe_name="d"),
        )
        warning = next(w for w in result.warnings if w.code == "FEW_OVERLAPPING_ROWS")
        assert sorted(warning.columns) == ["a", "b"]

    def test_full_overlap_is_quiet(self):
        rng = np.random.default_rng(9)
        frame = pd.DataFrame({"a": rng.normal(size=40), "b": rng.normal(size=40)})
        result = correlations(_state(frame), CorrelationsInput(dataframe_name="d"))
        assert result.warnings == []


class TestDetectOutliers:
    def test_zero_mad_does_not_mean_no_outliers(self):
        """A store with mostly zero-order days and one huge day."""
        frame = pd.DataFrame({"orders": [0] * 20 + [1000]})
        result = detect_outliers(
            _state(frame),
            DetectOutliersInput(dataframe_name="d", method="modified_zscore"),
        )
        assert result.outlier_summary["total_outliers_found"] == 0
        warning = next(w for w in result.warnings if w.code == "DEGENERATE_SPREAD")
        assert warning.columns == ["orders"]

    def test_zero_iqr_is_flagged_too(self):
        frame = pd.DataFrame({"orders": [0] * 20 + [1, 1000]})
        result = detect_outliers(_state(frame), DetectOutliersInput(dataframe_name="d"))
        assert "DEGENERATE_SPREAD" in _codes(result)

    def test_quality_report_shares_the_check(self):
        frame = pd.DataFrame({"orders": [0] * 20 + [1000]})
        result = data_quality_report(
            _state(frame),
            DataQualityReportInput(
                dataframe_name="d", outlier_method="modified_zscore"
            ),
        )
        assert "DEGENERATE_SPREAD" in _codes(result)


class TestTTest:
    def test_constant_samples_are_refused_not_called_insignificant(self):
        frame = pd.DataFrame({"a": [5.0] * 6, "b": [5.0] * 6})
        with pytest.raises(ValueError, match="undefined"):
            t_test(
                _state(frame),
                TTestInput(dataframe_name="d", column_a="a", column_b="b"),
            )

    def test_student_with_very_unequal_variances_is_flagged(self):
        rng = np.random.default_rng(10)
        a = rng.normal(0, 1, 60)
        b = np.full(60, np.nan)
        b[:12] = rng.normal(0, 10, 12)
        result = t_test(
            _state(pd.DataFrame({"a": a, "b": b})),
            TTestInput(dataframe_name="d", column_a="a", column_b="b"),
        )
        assert "UNEQUAL_VARIANCE" in _codes(result)

    def test_welch_is_quiet(self):
        rng = np.random.default_rng(10)
        a = rng.normal(0, 1, 60)
        b = np.full(60, np.nan)
        b[:12] = rng.normal(0, 10, 12)
        result = t_test(
            _state(pd.DataFrame({"a": a, "b": b})),
            TTestInput(dataframe_name="d", column_a="a", column_b="b", equal_var=False),
        )
        assert result.warnings == []


class TestZTest:
    def test_small_samples_without_a_known_sd_are_flagged(self):
        rng = np.random.default_rng(11)
        frame = pd.DataFrame({"a": rng.normal(0, 1, 8), "b": rng.normal(1, 1, 8)})
        result = z_test(
            _state(frame), ZTestInput(dataframe_name="d", column_a="a", column_b="b")
        )
        assert "SMALL_SAMPLE" in _codes(result)

    def test_known_population_sd_is_quiet(self):
        rng = np.random.default_rng(11)
        frame = pd.DataFrame({"a": rng.normal(0, 1, 8), "b": rng.normal(1, 1, 8)})
        result = z_test(
            _state(frame),
            ZTestInput(
                dataframe_name="d",
                column_a="a",
                column_b="b",
                population_std_a=1.0,
                population_std_b=1.0,
            ),
        )
        assert result.warnings == []


# =============================================================================
# Time series
# =============================================================================


def _fit(frame, **kwargs):
    state = _state(frame)
    return state, fit_arima(
        state,
        FitARIMAInput(
            dataframe_name="d",
            target_column="revenue",
            date_column="date",
            p=1,
            d=0,
            q=0,
            **kwargs,
        ),
    )


class TestArimaSeriesPreparation:
    def test_duplicate_dates_are_refused(self):
        """Order-level rows are not a daily series."""
        frame = pd.concat([_daily(30), _daily(30)]).reset_index(drop=True)
        _, result = _fit(frame)
        assert isinstance(result, OperationError)
        assert result.error_type == "DuplicateDates"

    def test_unsorted_dates_are_sorted_and_flagged(self):
        frame = _daily(60)
        _, ordered = _fit(frame)
        _, shuffled = _fit(frame.sample(frac=1, random_state=0).reset_index(drop=True))
        assert not isinstance(shuffled, OperationError)
        assert shuffled.aic == pytest.approx(ordered.aic)
        assert "UNSORTED_DATES" in _codes(shuffled)

    def test_gaps_are_flagged(self):
        frame = _daily(60).drop(index=[10, 11, 30]).reset_index(drop=True)
        state, result = _fit(frame)
        assert "IRREGULAR_SPACING" in _codes(result)
        assert "IRREGULAR_SPACING" in _logged(state)

    def test_a_clean_series_is_quiet(self):
        _, result = _fit(_daily(60))
        assert result.warnings == []

    def test_nulls_dropped_from_a_series_without_dates_are_reported(self):
        """Without dates, dropping a null closes the gap and nobody can tell."""
        frame = _daily(60)
        frame.loc[[5, 6, 20], "revenue"] = np.nan
        state = _state(frame)
        result = fit_arima(
            state,
            FitARIMAInput(dataframe_name="d", target_column="revenue", p=1, d=0, q=0),
        )
        warning = next(w for w in result.warnings if w.code == "NULLS_DROPPED")
        assert "3" in warning.message

    def test_stationarity_check_shares_the_guard(self):
        frame = pd.concat([_daily(30), _daily(30)]).reset_index(drop=True)
        result = check_stationarity(
            _state(frame),
            StationarityTestInput(
                dataframe_name="d", target_column="revenue", date_column="date"
            ),
        )
        assert isinstance(result, OperationError)
        assert result.error_type == "DuplicateDates"


class TestFindOptimalArima:
    def test_comparing_aic_across_differencing_orders_is_flagged(self):
        result = find_optimal_arima(
            _state(_daily(60)),
            FindOptimalARIMAInput(
                dataframe_name="d",
                target_column="revenue",
                date_column="date",
                max_p=1,
                max_q=1,
                max_d=1,
            ),
        )
        assert "AIC_ACROSS_D" in _codes(result)

    def test_fixed_d_is_quiet(self):
        result = find_optimal_arima(
            _state(_daily(60)),
            FindOptimalARIMAInput(
                dataframe_name="d",
                target_column="revenue",
                date_column="date",
                max_p=1,
                max_q=1,
                fixed_d=0,
            ),
        )
        assert result.warnings == []


class TestForecastArima:
    def test_a_horizon_beyond_the_cap_says_it_was_cut(self):
        state, fitted = _fit(_daily(60))
        result = forecast_arima(
            state,
            ForecastARIMAInput(
                model_id=fitted.model_id,
                forecast_number=2,
                forecast_unit="years",
                include_plot=False,
            ),
        )
        assert result.n_periods == 365
        assert "HORIZON_CAPPED" in _codes(result)


class TestInferFrequency:
    def test_unsorted_daily_dates_are_still_daily(self):
        frame = _daily(30).sample(frac=1, random_state=1).reset_index(drop=True)
        result = infer_frequency(
            _state(frame), InferFrequencyInput(dataframe_name="d", date_column="date")
        )
        assert result.frequency_days == 1.0


class TestTimeseriesWorkflowLiftsWarnings:
    def test_fit_warnings_reach_the_workflow_result(self):
        frame = _daily(80).drop(index=[10, 11, 30]).reset_index(drop=True)
        result = run_timeseries_forecast(
            _state(frame),
            RunTimeseriesForecastInput(
                dataframe_name="d",
                target_column="revenue",
                date_column="date",
                config=TimeSeriesConfig(
                    auto_find_params=False,
                    check_stationarity=False,
                    validate_dates=False,
                    generate_forecast_plot=False,
                ),
            ),
        )
        assert "IRREGULAR_SPACING" in [w.code for w in result.warnings]
