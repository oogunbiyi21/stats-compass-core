# Unattended-operation audit

Stage A runs these tools with nobody reading the output. The question asked of
each one is not "does it work" but **"what does it do when nobody checks the
output"**. A tool that fails loudly is fine. A tool that returns a plausible,
confident, wrong number is not.

**Scope.** Every cleaning, transform, EDA and time-series subtool. These are the
categories the Stage A analysis path touches. The ML trainers are not on that path
and are not listed. Parent tools (`describe_*`, `execute_*`) only dispatch, so
they are not listed either. `tests/test_unattended_audit.py` checks that the
table below names exactly the registered subtools in those categories (26 at the
time of writing) and that every status is one of the three below. A new tool
without a row fails that test.

## Statuses

| Status | Meaning |
|---|---|
| `guarded` | It refuses input it cannot handle correctly, **or** it cannot produce a wrong number by construction and its result states every change it made (rows dropped, values filled), so nothing it does is silent. |
| `warns` | It can return a result that may be wrong. When it detects the condition, it attaches a named `ToolWarning` (`code`, `message`, `columns`) to the result's `warnings` and records it in the op-log (`state.get_history()`, operation `"warning"`). It may also refuse the worst cases. |
| `not on path` | Not used by the Stage A analysis path. Answered anyway. |

Workflows lift every step's warnings into `WorkflowResult.warnings`.

## The sweep

| Tool | Area | Status | When nobody checks the output | Guard or warning |
|---|---|---|---|---|
| `apply_imputation` | cleaning | guarded | Fills missing values with a mean/median/mode/constant and reports how many it filled per column. The fill is what was asked for, but a mean-filled gap in a trending series is flat. Do not impute series on the analysis path; nulls arrive as nulls. | Counts in result |
| `clean_dates` | cleaning | warns | Used to coerce unreadable dates to missing silently, then forward-fill them with a neighbour's date, so orders moved to days they were not placed on. `create_missing_dates` silently did nothing whenever there were gaps to fill. | `UNPARSEABLE_VALUES`, `DATES_FILLED`, `GAPS_NOT_FILLED` |
| `convert_dtype` | cleaning | warns | Values that are neither a missing marker nor parseable become missing, and are counted and sampled. A column where nothing parses is refused (wrong type requested). | `UNPARSEABLE_VALUES`; refuses all-unparseable; `errors="raise"` |
| `dedupe` | cleaning | guarded | Removes duplicate rows and reports how many. Full-row dedupe also removes rows that are legitimately identical (two identical line items), so pass `subset` with a key. | Count in result |
| `drop_na` | cleaning | guarded | Drops rows or columns with missing values and reports how many. | Count in result |
| `handle_outliers` | cleaning | guarded | Caps, removes or transforms values as asked and reports the count and before/after statistics. Percentile capping always touches the top 1%, and capping promotion days erases the effect the promotion ITS measures. Not for the analysis path's series. | Count and stats in result |
| `bin_rare_categories` | transforms | not on path | ML feature engineering. An identifier column used to collapse to one label and reach the model as a constant (T6.7). | Warns `HIGH_CARDINALITY`, `NOT_CATEGORICAL` |
| `filter_dataframe` | transforms | guarded | Returns the matching rows; zero matches is an empty frame with its shape in the result, and a bad query raises. | Shape in result |
| `groupby_aggregate` | transforms | warns | Rows with a missing group key are left out of every group, so "revenue by discount code" lost every order without a code. A group whose values are all missing sums to 0. A `sum` of numbers stored as text joins the strings. | `NULL_GROUP_KEYS`, `NULL_GROUP_SUMMED`, `NUMERIC_AS_TEXT` |
| `mean_target_encoding` | transforms | not on path | ML feature engineering. Identifier columns are now skipped. The encoder is fitted on the whole frame before the trainer splits it (cross-fitted, so mitigated, but test rows still shape the encoding). | Warns `HIGH_CARDINALITY` |
| `pivot` | transforms | warns | Cells with more than one row are aggregated silently with `aggfunc` (default `mean`), so order-level rows pivoted to day × channel gave average order value where a total was expected. Rows with missing keys are dropped. | `CELLS_AGGREGATED`, `NULL_GROUP_KEYS` |
| `split_column_by_group` | transforms | guarded | Numeric group values used to match nothing and produce empty columns; groups are now matched as text. | Fixed |
| `analyze_missing_data` | eda | warns | Missing values spelled as text (`"null"`, `"nan"`, `""`) count as present, so a column of them got "Data quality looks good!". | `MISSING_AS_TEXT` |
| `chi_square_goodness_of_fit` | eda | warns | Reports low expected counts in `low_expected_warning` (a field that predates `ToolWarning`). Caller-supplied expected frequencies are matched to categories by sorted position, not by name. | `low_expected_warning` |
| `chi_square_independence` | eda | warns | As above, for the contingency table. | `low_expected_warning` |
| `correlations` | eda | warns | With no `min_periods`, a pair overlapping on two or three rows reports a coefficient near ±1. | `FEW_OVERLAPPING_ROWS` (< 10 shared rows) |
| `data_quality_report` | eda | warns | Same blind spot as `analyze_missing_data`, plus the outlier check below; the quality score is inflated by both. | `MISSING_AS_TEXT`, `DEGENERATE_SPREAD` |
| `describe` | eda | warns | A numeric column stored as text gets no numeric statistics and nothing says it was left out. | `NUMERIC_AS_TEXT` |
| `detect_outliers` | eda | warns | When more than half the values are identical (mostly zero-order days), MAD is 0 and the modified z-score reports no outliers however extreme the rest. When IQR is 0, everything off the common value is flagged. It does not account for trend or season: a December peak is an "outlier". | `DEGENERATE_SPREAD` |
| `t_test` | eda | warns | Two constant samples gave a NaN p-value, which reads as "not significant"; now refused. Student's test with very unequal variances gives an unreliable p-value. It assumes independent observations: day-level samples from a series are autocorrelated and the p-value is too small. That is **not detected** (see below). | `UNEQUAL_VARIANCE` (ratio > 4); refuses constant samples |
| `z_test` | eda | warns | With standard deviations estimated from fewer than 30 rows, the p-value is too small. | `SMALL_SAMPLE` (n < 30, no known SD) |
| `check_stationarity` | time series | warns | Tested rows in the order given. With the new optional `date_column` it uses the shared series guard: duplicate dates are refused, unsorted rows are sorted, and gaps are named. Without `date_column`, row order is trusted. | `UNSORTED_DATES`, `IRREGULAR_SPACING`, `NULLS_DROPPED`; refuses duplicate dates |
| `find_optimal_arima` | time series | warns | Ranked models with different differencing orders by AIC/BIC, which is not comparable across d. Shares the series guard. | `AIC_ACROSS_D` (when `fixed_d` is unset), plus the series guard |
| `fit_arima` | time series | warns | Fitted order-level rows, unsorted rows and gappy series as if each were one evenly spaced value per period; a shuffled series fitted a different model without complaint. | Refuses duplicate dates; `UNSORTED_DATES` (sorted), `IRREGULAR_SPACING`, `NULLS_DROPPED` |
| `forecast_arima` | time series | warns | Silently stopped at 365 steps while the message named the full requested horizon. Intervals are the fitted model's and assume it is right. | `HORIZON_CAPPED` |
| `infer_frequency` | time series | guarded | Measured spacing in row order, so shuffled daily dates came out as −1 day. It now measures between distinct dates in order. | Fixed |

## Thresholds chosen

These are module constants, not arguments. They are guard thresholds, not
verdict calibration; if hosted needs to tune one, it should become an argument
with this value as its default.

| Constant | Value | Where |
|---|---|---|
| Identifier cardinality | > 200 unique, or > 50% of non-null rows | `transforms/_cardinality.py` |
| Leakage: minimum training rows | 20 | `ml/leakage.py` |
| Leakage: rank correlation treated as the target | ≥ 0.999 | `ml/leakage.py` |
| Correlation overlap floor | 10 rows | `eda/correlations.py` |
| Student variance ratio | 4 | `eda/hypothesis_tests.py` |
| z-test sample floor | 30 | `eda/hypothesis_tests.py` |
| "Numbers as text" | ≥ 90% of non-missing values parse | `utils/text_values.py` |
| Forecast cap | 365 steps (pre-existing) | `ml/timeseries/arima.py` |

## Known and not guarded

Recorded so they are not mistaken for covered.

- **Autocorrelation in `t_test`/`z_test`.** Day-level samples are not
  independent, so period-over-period p-values on them are too small. Row order
  is not reliably time order inside these tools, so a check here would misfire.
  The Stage A period comparison belongs in core's verdict procedure, which must
  account for it.
- **Numeric identifiers in the trainers' fallback.** An all-unique integer
  `customer_id` is still swept in when `feature_columns` is not declared. That
  path now carries `FEATURES_INFERRED`. ML is off the Stage A path.
- **Seasonality in `detect_outliers`.** Peaks that are seasonal are flagged as
  outliers. Seasonal decomposition is new work, not an audit fix.
- **MCP summaries.** `stats-compass-mcp`'s workflow summary forwards each step's
  `result` (so step-level warnings reach the agent) but not the top-level
  `WorkflowResult.warnings`.
