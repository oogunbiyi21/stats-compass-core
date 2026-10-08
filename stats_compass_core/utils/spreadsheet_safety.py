"""
Spreadsheet Safety Utilities.

Provides protection against spreadsheet formula injection attacks 
(also known as CSV injection or formula injection).

When spreadsheet files (CSV, XLSX, XLS) are opened in applications 
like Excel, Google Sheets, or LibreOffice Calc, cells beginning with 
certain characters (=, +, -, @, \t, \r, \n) may be interpreted as 
formulas and executed, potentially leading to:
- Remote code execution
- Data exfiltration via web requests
- Information disclosure

This module sanitizes DataFrame values before spreadsheet export to 
prevent these attacks.

References:
- OWASP: https://owasp.org/www-community/attacks/CSV_Injection
- CWE-1236: https://cwe.mitre.org/data/definitions/1236.html
"""

import pandas as pd

# Characters that trigger formula interpretation in spreadsheet applications
FORMULA_TRIGGER_CHARS = frozenset({"=", "+", "-", "@", "\t", "\r", "\n"})

# Prefix that neutralizes formula interpretation (single quote)
SAFE_PREFIX = "'"


def sanitize_cell(value: str) -> str:
    """
    Sanitize a single cell value to prevent CSV injection.
    
    If the value starts with a formula trigger character, prepend
    a single quote which causes spreadsheets to treat it as text.
    
    Args:
        value: The cell value to sanitize.
        
    Returns:
        The sanitized value.
        
    Examples:
        >>> sanitize_cell("=SUM(A1:A10)")
        "'=SUM(A1:A10)"
        >>> sanitize_cell("+cmd|'/C calc'!A0")
        "'+cmd|'/C calc'!A0"
        >>> sanitize_cell("Normal text")
        "Normal text"
    """
    if value and value[0] in FORMULA_TRIGGER_CHARS:
        return f"{SAFE_PREFIX}{value}"
    return value


def sanitize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sanitize everything a CSV export writes as text, to prevent CSV injection.
    
    Creates a copy of the DataFrame in which every string that starts with a
    formula trigger character is prefixed with a single quote: cells in object,
    ``string`` and ``category`` columns, column headers, row labels and index
    names. Only object columns were covered before; a header or a typed text
    column reached the file as written (security scan F9, 8 Oct 2026).
    
    Numeric values and missing values are left unchanged. A ``category``
    column comes back as object, since quoting can merge or split categories.
    
    Args:
        df: The DataFrame to sanitize.
        
    Returns:
        A new DataFrame with sanitized string values and labels.
        
    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "name": ["Alice", "=HYPERLINK('http://evil.com')"],
        ...     "score": [100, 200]
        ... })
        >>> safe_df = sanitize_dataframe(df)
        >>> safe_df["name"].iloc[1]
        "'=HYPERLINK('http://evil.com')"
        >>> safe_df["score"].iloc[1]  # Non-string unchanged
        200
    """
    # Work on a copy to avoid modifying the original
    df_safe = df.copy()

    # Cells, by position so repeated column names are handled too
    for i in range(df_safe.shape[1]):
        column = df_safe.iloc[:, i]
        dtype = column.dtype
        if dtype == "object":
            df_safe.isetitem(i, column.apply(_sanitize_label))
        elif isinstance(dtype, pd.StringDtype):
            df_safe.isetitem(i, column.astype(object).apply(_sanitize_label).astype(dtype))
        elif isinstance(dtype, pd.CategoricalDtype):
            df_safe.isetitem(i, column.astype(object).apply(_sanitize_label))

    # Labels: headers, row labels (written when index=True) and their names
    df_safe.columns = _sanitize_index(df_safe.columns)
    df_safe.index = _sanitize_index(df_safe.index)

    return df_safe


def _sanitize_label(value):
    """A string, or each string in a tuple label; anything else unchanged."""
    if isinstance(value, str):
        return sanitize_cell(value)
    if isinstance(value, tuple):
        return tuple(_sanitize_label(v) for v in value)
    return value


def _sanitize_index(index: pd.Index) -> pd.Index:
    names = [_sanitize_label(n) for n in index.names]
    if isinstance(index, pd.MultiIndex) or index.dtype == "object" or isinstance(
        index.dtype, (pd.StringDtype, pd.CategoricalDtype)
    ):
        index = index.map(_sanitize_label)
    return index.set_names(names)
