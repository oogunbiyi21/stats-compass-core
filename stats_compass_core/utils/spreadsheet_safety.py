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
    Sanitize all string values in a DataFrame to prevent CSV injection.
    
    Creates a copy of the DataFrame with all string cells that start
    with formula trigger characters prefixed with a single quote.
    
    Non-string columns are left unchanged.
    
    Args:
        df: The DataFrame to sanitize.
        
    Returns:
        A new DataFrame with sanitized string values.
        
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
    
    # Only process object (string) columns
    for col in df_safe.columns:
        if df_safe[col].dtype == "object":
            # Apply sanitization only to string values
            df_safe[col] = df_safe[col].apply(
                lambda x: sanitize_cell(x) if isinstance(x, str) else x
            )
    
    return df_safe
