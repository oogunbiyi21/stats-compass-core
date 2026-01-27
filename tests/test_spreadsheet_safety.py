"""Tests for spreadsheet formula injection protection."""

import pandas as pd
import pytest

from stats_compass_core.utils.spreadsheet_safety import (
    FORMULA_TRIGGER_CHARS,
    sanitize_cell,
    sanitize_dataframe,
)


class TestSanitizeCell:
    """Tests for sanitize_cell function."""

    def test_formula_with_equals(self):
        """Cells starting with = should be prefixed."""
        assert sanitize_cell("=SUM(A1:A10)") == "'=SUM(A1:A10)"
        assert sanitize_cell("=cmd|'/C calc'!A0") == "'=cmd|'/C calc'!A0"
        assert sanitize_cell("=HYPERLINK('http://evil.com')") == "'=HYPERLINK('http://evil.com')"

    def test_formula_with_plus(self):
        """Cells starting with + should be prefixed."""
        assert sanitize_cell("+cmd|'/C calc'!A0") == "'+cmd|'/C calc'!A0"
        assert sanitize_cell("+1") == "'+1"

    def test_formula_with_minus(self):
        """Cells starting with - should be prefixed."""
        assert sanitize_cell("-cmd|'/C calc'!A0") == "'-cmd|'/C calc'!A0"
        assert sanitize_cell("-1") == "'-1"

    def test_formula_with_at(self):
        """Cells starting with @ should be prefixed."""
        assert sanitize_cell("@SUM(1+1)*cmd|'/C calc'!A0") == "'@SUM(1+1)*cmd|'/C calc'!A0"

    def test_formula_with_tab(self):
        """Cells starting with tab should be prefixed."""
        assert sanitize_cell("\t=cmd") == "'\t=cmd"

    def test_formula_with_newlines(self):
        """Cells starting with newlines should be prefixed."""
        assert sanitize_cell("\r=cmd") == "'\r=cmd"
        assert sanitize_cell("\n=cmd") == "'\n=cmd"

    def test_normal_text_unchanged(self):
        """Normal text should not be modified."""
        assert sanitize_cell("Normal text") == "Normal text"
        assert sanitize_cell("Hello World") == "Hello World"
        assert sanitize_cell("123") == "123"
        assert sanitize_cell("email@example.com") == "email@example.com"  # @ not at start

    def test_empty_string(self):
        """Empty strings should be unchanged."""
        assert sanitize_cell("") == ""

    def test_equals_in_middle(self):
        """Equals sign in middle of string should be unchanged."""
        assert sanitize_cell("a=b") == "a=b"
        assert sanitize_cell("x + y = z") == "x + y = z"


class TestSanitizeDataFrame:
    """Tests for sanitize_dataframe function."""

    def test_sanitizes_string_columns(self):
        """String columns should be sanitized."""
        df = pd.DataFrame({
            "name": ["Alice", "=HYPERLINK('http://evil.com')"],
            "note": ["Normal", "+malicious"],
        })
        safe_df = sanitize_dataframe(df)
        
        assert safe_df["name"].iloc[0] == "Alice"
        assert safe_df["name"].iloc[1] == "'=HYPERLINK('http://evil.com')"
        assert safe_df["note"].iloc[0] == "Normal"
        assert safe_df["note"].iloc[1] == "'+malicious"

    def test_preserves_numeric_columns(self):
        """Numeric columns should be unchanged."""
        df = pd.DataFrame({
            "text": ["=formula", "normal"],
            "number": [100, -200],
            "float": [1.5, -2.5],
        })
        safe_df = sanitize_dataframe(df)
        
        assert safe_df["number"].iloc[0] == 100
        assert safe_df["number"].iloc[1] == -200
        assert safe_df["float"].iloc[0] == 1.5
        assert safe_df["float"].iloc[1] == -2.5

    def test_preserves_none_values(self):
        """None values in string columns should remain None."""
        df = pd.DataFrame({
            "text": ["=formula", None, "normal"],
        })
        safe_df = sanitize_dataframe(df)
        
        assert safe_df["text"].iloc[0] == "'=formula"
        assert pd.isna(safe_df["text"].iloc[1])
        assert safe_df["text"].iloc[2] == "normal"

    def test_does_not_modify_original(self):
        """Original DataFrame should not be modified."""
        df = pd.DataFrame({
            "text": ["=formula"],
        })
        original_value = df["text"].iloc[0]
        
        safe_df = sanitize_dataframe(df)
        
        assert df["text"].iloc[0] == original_value  # Original unchanged
        assert safe_df["text"].iloc[0] == "'=formula"  # New is sanitized

    def test_empty_dataframe(self):
        """Empty DataFrame should work."""
        df = pd.DataFrame()
        safe_df = sanitize_dataframe(df)
        assert safe_df.empty

    def test_mixed_types_in_object_column(self):
        """Object columns with mixed types should only sanitize strings."""
        df = pd.DataFrame({
            "mixed": ["=formula", 123, None, "normal", 45.6],
        })
        safe_df = sanitize_dataframe(df)
        
        assert safe_df["mixed"].iloc[0] == "'=formula"
        assert safe_df["mixed"].iloc[1] == 123  # int unchanged
        assert pd.isna(safe_df["mixed"].iloc[2])  # None unchanged
        assert safe_df["mixed"].iloc[3] == "normal"
        assert safe_df["mixed"].iloc[4] == 45.6  # float unchanged
