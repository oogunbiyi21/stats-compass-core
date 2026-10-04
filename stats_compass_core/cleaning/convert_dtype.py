"""
Tool for converting text columns to numeric, datetime or boolean.

Exports carry numbers as text, with missing values spelled "nan", "null", ""
and so on. pandas reads those as strings, so every numeric tool downstream
either refuses the column or, worse, a hand-rolled coercion turns real data
into NaN without saying how much.

A value that is neither a recognised missing marker nor parseable becomes
missing by default, and the result counts it and shows examples. A column where
nothing parses is refused: that is a request for the wrong type, not dirty data.
"""

from typing import Literal

import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import (
    ColumnConversion,
    ConvertDtypeResult,
    ToolWarning,
)
from stats_compass_core.state import DataFrameState
from stats_compass_core.utils.text_values import NULL_TOKENS as DEFAULT_NULL_TOKENS

TRUE_TOKENS = {"true", "t", "yes", "y", "1", "1.0"}
FALSE_TOKENS = {"false", "f", "no", "n", "0", "0.0"}

SAMPLE_SIZE = 5


class ConvertDtypeInput(StrictToolInput):
    """Input schema for convert_dtype tool."""

    dataframe_name: str | None = Field(
        default=None, description="Name of DataFrame to operate on. Uses active if not specified."
    )
    columns: list[str] = Field(min_length=1, description="Columns to convert")
    to: Literal["numeric", "datetime", "bool"] = Field(
        description="Target type: 'numeric' (float/int), 'datetime', or 'bool'"
    )
    null_tokens: list[str] = Field(
        default_factory=lambda: list(DEFAULT_NULL_TOKENS),
        description=(
            "Text that means 'missing'. Matched after trimming whitespace and "
            "ignoring case. Default covers '', 'nan', 'null', 'none', 'na', "
            "'n/a', 'nat' and '-'."
        ),
    )
    errors: Literal["coerce", "raise"] = Field(
        default="coerce",
        description=(
            "What to do with a value that is not a missing marker and does not "
            "parse. 'coerce' makes it missing and reports how many and which; "
            "'raise' refuses and leaves the DataFrame unchanged."
        ),
    )
    datetime_format: str | None = Field(
        default=None,
        description=(
            "strftime format for 'datetime', e.g. '%d/%m/%Y'. Inferred if omitted."
        ),
    )
    dayfirst: bool = Field(
        default=False,
        description="For 'datetime' without a format: read 01/02 as 1 February",
    )
    save_as: str | None = Field(
        default=None,
        description="Save result as new DataFrame with this name. If not provided, modifies in place.",
    )


def _strip_null_tokens(series: pd.Series, tokens: set[str]) -> tuple[pd.Series, int]:
    """Replace missing-value spellings with NA. Returns (series, how many)."""
    if not (series.dtype == "object" or pd.api.types.is_string_dtype(series)):
        return series, 0
    text = series.map(lambda v: v.strip().lower() if isinstance(v, str) else v)
    is_token = text.isin(tokens) & series.notna()
    return series.mask(is_token), int(is_token.sum())


def _to_bool(series: pd.Series) -> pd.Series:
    def parse(value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in TRUE_TOKENS:
            return True
        if text in FALSE_TOKENS:
            return False
        return pd.NA

    return series.map(parse).astype("boolean")


def _convert(series: pd.Series, params: ConvertDtypeInput) -> pd.Series:
    if params.to == "numeric":
        is_bool = pd.api.types.is_bool_dtype(series)
        if pd.api.types.is_numeric_dtype(series) and not is_bool:
            return series
        return pd.to_numeric(series, errors="coerce")
    if params.to == "datetime":
        if pd.api.types.is_datetime64_any_dtype(series):
            return series
        return pd.to_datetime(
            series,
            errors="coerce",
            format=params.datetime_format,
            dayfirst=params.dayfirst,
        )
    return _to_bool(series)


@registry.register(
    category="cleaning",
    input_schema=ConvertDtypeInput,
    description=(
        "Convert text columns to numeric, datetime or bool, treating 'nan', "
        "'null', '' etc. as missing and reporting values that do not parse"
    ),
)
def convert_dtype(
    state: DataFrameState, params: ConvertDtypeInput
) -> ConvertDtypeResult:
    """
    Convert columns to numeric, datetime or boolean.

    Args:
        state: DataFrameState containing the DataFrame to operate on
        params: Parameters for the conversion

    Returns:
        ConvertDtypeResult with per-column counts of missing markers and
        unparseable values

    Raises:
        ValueError: If a column does not exist, if errors='raise' and a value
            does not parse, or if no value in a column parses
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()

    missing_cols = [c for c in params.columns if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Columns not found: {missing_cols}. Available columns: {list(df.columns)}"
        )

    tokens = {t.strip().lower() for t in params.null_tokens}
    result_df = df.copy()
    details: dict[str, ColumnConversion] = {}
    warnings: list[ToolWarning] = []

    # Convert everything before writing anything, so a refusal leaves the
    # DataFrame as it was.
    for col in params.columns:
        original = df[col]
        cleaned, n_tokens = _strip_null_tokens(original, tokens)
        converted = _convert(cleaned, params)

        failed = cleaned.notna() & converted.isna()
        n_failed = int(failed.sum())
        sample = [str(v) for v in pd.unique(cleaned[failed])[:SAMPLE_SIZE]]
        n_values = int(cleaned.notna().sum())

        if n_failed and n_failed == n_values:
            raise ValueError(
                f"Column '{col}': none of its {n_values} non-missing values "
                f"converts to {params.to} (e.g. {sample}). Check the column and "
                f"the requested type."
            )
        if n_failed and params.errors == "raise":
            raise ValueError(
                f"Column '{col}': {n_failed} value(s) do not convert to "
                f"{params.to}, e.g. {sample}. Nothing was changed."
            )

        if n_failed:
            warnings.append(ToolWarning(
                code="UNPARSEABLE_VALUES",
                columns=[col],
                message=(
                    f"{n_failed} of {n_values} value(s) in '{col}' did not convert "
                    f"to {params.to} and are now missing, e.g. {sample}."
                ),
            ))

        result_df[col] = converted
        details[col] = ColumnConversion(
            from_dtype=str(original.dtype),
            to_dtype=str(converted.dtype),
            null_tokens=n_tokens,
            unparseable=n_failed,
            unparseable_sample=sample,
        )

    result_name = params.save_as or source_name
    stored_name = state.set_dataframe(
        result_df, name=result_name, operation="convert_dtype"
    )
    for warning in warnings:
        state.record_warning(
            warning.code, stored_name, warning.message, warning.columns
        )

    total_failed = sum(d.unparseable for d in details.values())
    total_tokens = sum(d.null_tokens for d in details.values())
    message = (
        f"Converted {len(details)} column(s) to {params.to}. "
        f"{total_tokens} missing marker(s) became missing."
    )
    if total_failed:
        message += f" {total_failed} unparseable value(s) also became missing."

    return ConvertDtypeResult(
        success=True,
        dataframe_name=stored_name,
        source_dataframe=source_name,
        to=params.to,
        columns=details,
        message=message,
        warnings=warnings,
    )
