"""
Tool for cleaning and validating date columns in DataFrames.
"""

from __future__ import annotations

import pandas as pd
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.results import DataFrameMutationResult
from stats_compass_core.state import DataFrameState


class CleanDatesInput(StrictToolInput):
    """Input schema for clean_dates tool."""

    dataframe_name: str | None = Field(
        default=None,
        description="Name of DataFrame to operate on. Uses active if not specified.",
    )
    date_column: str = Field(
        description="Name of the date column to clean",
    )
    fill_method: str = Field(
        default="ffill",
        pattern="^(ffill|bfill|interpolate|drop)$",
        description="Method to handle missing dates: ffill (forward fill), bfill (backward fill), interpolate, or drop",
    )
    infer_frequency: bool = Field(
        default=True,
        description="Whether to infer the frequency of the date column and create missing dates to complete the sequence",
    )
    create_missing_dates: bool = Field(
        default=False,
        description="Whether to create missing dates in the sequence (e.g., fill gaps in daily data)",
    )
    save_as: str | None = Field(
        default=None,
        description="Save result as new DataFrame with this name. If not provided, modifies in place.",
    )


def validate_date_column(
    df: pd.DataFrame,
    date_column: str,
    check_nulls: bool = True,
    check_duplicates: bool = True,
    check_chronological: bool = True,
    check_gaps: bool = False,
) -> dict[str, any]:
    """
    Validate a date column for time series requirements.
    
    Args:
        df: DataFrame containing the date column
        date_column: Name of the date column to validate
        check_nulls: Whether to check for null values
        check_duplicates: Whether to check for duplicate dates
        check_chronological: Whether to verify chronological order
        check_gaps: Whether to check for gaps in the date sequence
    
    Returns:
        Dictionary with validation results:
        - is_valid: bool
        - errors: list of error messages
        - warnings: list of warning messages
        - null_count: number of null values
        - duplicate_count: number of duplicate dates
        - is_chronological: whether dates are in order
        - inferred_frequency: inferred frequency string (if applicable)
        - gaps_count: number of gaps in the sequence
    """
    result = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "null_count": 0,
        "duplicate_count": 0,
        "is_chronological": True,
        "inferred_frequency": None,
        "gaps_count": 0,
    }
    
    if date_column not in df.columns:
        result["is_valid"] = False
        result["errors"].append(f"Date column '{date_column}' not found in DataFrame")
        return result
    
    date_series = df[date_column]
    
    # Check nulls
    if check_nulls:
        null_count = int(date_series.isna().sum())
        result["null_count"] = null_count
        if null_count > 0:
            result["is_valid"] = False
            result["errors"].append(f"Found {null_count} null values in date column ({null_count/len(df)*100:.2f}%)")
    
    # Get non-null dates for further checks
    valid_dates = date_series.dropna()
    
    if len(valid_dates) == 0:
        result["is_valid"] = False
        result["errors"].append("No valid dates found in column")
        return result
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(valid_dates):
        try:
            valid_dates = pd.to_datetime(valid_dates)
        except Exception as e:
            result["is_valid"] = False
            result["errors"].append(f"Failed to parse dates: {str(e)}")
            return result
    
    # Check duplicates
    if check_duplicates:
        duplicate_count = int(valid_dates.duplicated().sum())
        result["duplicate_count"] = duplicate_count
        if duplicate_count > 0:
            result["warnings"].append(f"Found {duplicate_count} duplicate dates")
    
    # Check chronological order
    if check_chronological and len(valid_dates) > 1:
        is_sorted = valid_dates.is_monotonic_increasing
        result["is_chronological"] = is_sorted
        if not is_sorted:
            result["warnings"].append("Dates are not in chronological order")
    
    # Infer frequency and check gaps
    if check_gaps and len(valid_dates) > 2:
        try:
            sorted_dates = valid_dates.sort_values()
            freq = pd.infer_freq(sorted_dates)
            result["inferred_frequency"] = freq
            
            if freq:
                # Create expected date range
                expected_range = pd.date_range(
                    start=sorted_dates.iloc[0],
                    end=sorted_dates.iloc[-1],
                    freq=freq
                )
                gaps_count = len(expected_range) - len(sorted_dates)
                result["gaps_count"] = gaps_count
                
                if gaps_count > 0:
                    result["warnings"].append(f"Found {gaps_count} gaps in date sequence (expected {freq} frequency)")
        except Exception:
            result["warnings"].append("Could not infer date frequency")
    
    return result


@registry.register(
    category="cleaning",
    input_schema=CleanDatesInput,
    description="Clean and validate date columns by handling missing values, filling gaps, and ensuring proper date sequence",
)
def clean_dates(
    state: DataFrameState, params: CleanDatesInput
) -> DataFrameMutationResult:
    """
    Clean and validate a date column in a DataFrame.
    
    Handles:
    - Missing/null dates using ffill, bfill, interpolate, or drop
    - Optionally creates missing dates to complete the sequence
    - Validates date format and chronological order
    
    Args:
        state: DataFrameState containing the DataFrame to operate on
        params: Date cleaning parameters
    
    Returns:
        DataFrameMutationResult with summary of changes
    
    Raises:
        ValueError: If date column is missing or invalid
    """
    df = state.get_dataframe(params.dataframe_name)
    source_name = params.dataframe_name or state.get_active_dataframe_name()
    rows_before = len(df)
    
    date_column = params.date_column
    
    if date_column not in df.columns:
        raise ValueError(f"Date column '{date_column}' not found in DataFrame")
    
    cleaned_df = df.copy()
    changes_made = []
    rows_dropped = 0
    dates_filled = 0
    dates_created = 0
    
    # Convert to datetime if not already
    if not pd.api.types.is_datetime64_any_dtype(cleaned_df[date_column]):
        try:
            cleaned_df[date_column] = pd.to_datetime(cleaned_df[date_column], errors='coerce')
            changes_made.append("Converted to datetime format")
        except Exception as e:
            raise ValueError(f"Failed to parse date column: {str(e)}")
    
    # Count initial nulls
    initial_nulls = int(cleaned_df[date_column].isna().sum())
    
    # Handle missing dates
    if initial_nulls > 0:
        if params.fill_method == "drop":
            cleaned_df = cleaned_df.dropna(subset=[date_column])
            rows_dropped = rows_before - len(cleaned_df)
            changes_made.append(f"Dropped {rows_dropped} rows with missing dates")
        elif params.fill_method == "ffill":
            cleaned_df[date_column] = cleaned_df[date_column].ffill()
            dates_filled = initial_nulls - int(cleaned_df[date_column].isna().sum())
            changes_made.append(f"Forward filled {dates_filled} missing dates")
        elif params.fill_method == "bfill":
            cleaned_df[date_column] = cleaned_df[date_column].bfill()
            dates_filled = initial_nulls - int(cleaned_df[date_column].isna().sum())
            changes_made.append(f"Backward filled {dates_filled} missing dates")
        elif params.fill_method == "interpolate":
            # For interpolation, we need numeric timestamps
            cleaned_df[date_column] = pd.to_datetime(
                cleaned_df[date_column].astype('int64').interpolate()
            )
            dates_filled = initial_nulls - int(cleaned_df[date_column].isna().sum())
            changes_made.append(f"Interpolated {dates_filled} missing dates")
    
    # Create missing dates in sequence if requested
    if params.create_missing_dates and params.infer_frequency:
        valid_dates = cleaned_df[date_column].dropna()
        
        if len(valid_dates) > 2:
            try:
                # Sort by date and infer frequency
                cleaned_df = cleaned_df.sort_values(by=date_column)
                freq = pd.infer_freq(valid_dates)
                
                if freq:
                    # Create complete date range
                    full_date_range = pd.date_range(
                        start=valid_dates.min(),
                        end=valid_dates.max(),
                        freq=freq
                    )
                    
                    # Create DataFrame with full date range
                    full_df = pd.DataFrame({date_column: full_date_range})
                    
                    # Merge with original data
                    cleaned_df = full_df.merge(
                        cleaned_df,
                        on=date_column,
                        how='left'
                    )
                    
                    dates_created = len(cleaned_df) - rows_before
                    if dates_created > 0:
                        changes_made.append(f"Created {dates_created} missing dates to complete {freq} sequence")
            except Exception as e:
                changes_made.append(f"Could not create missing dates: {str(e)}")
    
    # Sort by date for chronological order
    cleaned_df = cleaned_df.sort_values(by=date_column).reset_index(drop=True)
    
    # Determine result name
    result_name = params.save_as if params.save_as else source_name
    
    # Save DataFrame to state
    state.set_dataframe(cleaned_df, name=result_name, operation="clean_dates")
    
    # Build summary message
    if not changes_made:
        message = f"Date column '{date_column}' is already clean (no changes needed)"
    else:
        message = f"Cleaned date column '{date_column}': " + "; ".join(changes_made)
    
    return DataFrameMutationResult(
        success=True,
        rows_before=rows_before,
        rows_after=len(cleaned_df),
        rows_affected=dates_filled + dates_created + rows_dropped,
        dataframe_name=result_name,
        operation="clean_dates",
        message=message,
        columns_affected=[date_column],
    )
