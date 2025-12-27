"""
File Safety Utilities.

Provides safe file operations that prevent accidental overwrites
and validate paths to avoid dangerous operations.
"""

import os
from pathlib import Path


class UnsafePathError(Exception):
    """Raised when a path is deemed unsafe (e.g., system directories, sensitive locations)."""
    pass


# Paths that should never be written to
FORBIDDEN_PATHS = {
    "/",
    "/bin",
    "/sbin",
    "/usr",
    "/usr/bin",
    "/usr/sbin",
    "/usr/local/bin",
    "/etc",
    "/var",
    "/System",
    "/Library",
    "/Applications",
    # Windows equivalents
    "C:\\Windows",
    "C:\\Windows\\System32",
    "C:\\Program Files",
    "C:\\Program Files (x86)",
}

# File extensions that should never be overwritten
PROTECTED_EXTENSIONS = {
    ".py",      # Python source
    ".pyx",     # Cython source
    ".pyi",     # Python stubs
    ".pyc",     # Compiled Python
    ".pyo",     # Optimized Python
    ".js",      # JavaScript
    ".ts",      # TypeScript
    ".jsx",     # React JSX
    ".tsx",     # React TSX
    ".sh",      # Shell scripts
    ".bash",    # Bash scripts
    ".zsh",     # Zsh scripts
    ".yml",     # YAML config
    ".yaml",    # YAML config
    ".toml",    # TOML config
    ".json",    # JSON (could be config)
    ".env",     # Environment variables
    ".gitignore",
    ".gitattributes",
    ".dockerignore",
    "Dockerfile",
    "Makefile",
    ".sql",     # SQL scripts
    ".rs",      # Rust
    ".go",      # Go
    ".java",    # Java
    ".c",       # C
    ".cpp",     # C++
    ".h",       # C headers
    ".hpp",     # C++ headers
    ".rb",      # Ruby
    ".swift",   # Swift
    ".kt",      # Kotlin
    ".md",      # Markdown docs
    ".rst",     # ReStructuredText docs
}

# Safe extensions for data output
SAFE_OUTPUT_EXTENSIONS = {
    ".csv",
    ".tsv",
    ".xlsx",
    ".xls",
    ".parquet",
    ".feather",
    ".arrow",
    ".joblib",
    ".pkl",
    ".pickle",
    ".png",
    ".jpg",
    ".jpeg",
    ".svg",
    ".pdf",
    ".html",
    ".txt",
    ".log",
}


def is_path_safe(filepath: str) -> tuple[bool, str | None]:
    """
    Check if a path is safe to write to.
    
    Args:
        filepath: The path to validate
        
    Returns:
        Tuple of (is_safe, error_message)
        If safe, error_message is None
    """
    # Expand and resolve path
    expanded = os.path.expanduser(filepath)
    resolved = os.path.abspath(expanded)
    path = Path(resolved)
    
    # Check for forbidden parent directories
    for forbidden in FORBIDDEN_PATHS:
        forbidden_path = Path(forbidden)
        try:
            # Check if resolved path is under a forbidden directory
            if forbidden_path.exists() and resolved.startswith(str(forbidden_path) + os.sep):
                # Allow if it's deep enough (user subdirectory)
                relative = path.relative_to(forbidden_path)
                # Must be at least 2 levels deep to be considered safe
                if len(relative.parts) < 2:
                    return False, f"Cannot write to system directory: {forbidden}"
        except (ValueError, OSError):
            pass
    
    # Check file extension
    suffix = path.suffix.lower()
    name = path.name.lower()
    
    # Check if it's a protected file by extension or name
    if suffix in PROTECTED_EXTENSIONS or name in PROTECTED_EXTENSIONS:
        return False, f"Cannot overwrite source/config files ({suffix or name}). Use a different extension like .csv, .joblib, .png"
    
    # Warn if not a typical data output extension
    if suffix and suffix not in SAFE_OUTPUT_EXTENSIONS:
        # Allow but warn (caller can decide)
        pass
    
    return True, None


def get_unique_filepath(filepath: str) -> str:
    """
    Get a unique filepath by adding a numeric suffix if the file exists.
    
    Examples:
        output.csv -> output.csv (if doesn't exist)
        output.csv -> output_1.csv (if output.csv exists)
        output.csv -> output_2.csv (if output.csv and output_1.csv exist)
    
    Args:
        filepath: The desired filepath
        
    Returns:
        A filepath that doesn't exist (either original or with _N suffix)
    """
    path = Path(filepath)
    
    if not path.exists():
        return filepath
    
    base = path.stem
    ext = path.suffix
    parent = path.parent
    
    counter = 1
    while True:
        candidate = parent / f"{base}_{counter}{ext}"
        if not candidate.exists():
            return str(candidate)
        counter += 1


def safe_write_path(
    filepath: str,
    create_dirs: bool = True,
) -> str:
    """
    Validate and prepare a path for safe writing.
    
    Never overwrites existing files - automatically adds numeric suffix
    (e.g., output_1.csv, output_2.csv) if file exists.
    
    Args:
        filepath: The target path
        create_dirs: If True, create parent directories if they don't exist
        
    Returns:
        The resolved absolute path (may have _N suffix if original existed)
        
    Raises:
        UnsafePathError: If the path is in a forbidden location or has a protected extension
    """
    # Expand and resolve
    expanded = os.path.expanduser(filepath)
    resolved = os.path.abspath(expanded)
    
    # Check safety
    is_safe, error = is_path_safe(resolved)
    if not is_safe:
        raise UnsafePathError(error)
    
    # Get unique path (auto-increment if exists)
    resolved = get_unique_filepath(resolved)
    
    # Create directories if needed
    if create_dirs:
        parent = os.path.dirname(resolved)
        if parent:
            os.makedirs(parent, exist_ok=True)
    
    return resolved


def safe_save_figure(
    fig,
    save_path: str | None,
    **savefig_kwargs,
) -> str | None:
    """
    Safely save a matplotlib figure to disk.
    
    Never overwrites - automatically adds numeric suffix if file exists.
    
    Args:
        fig: Matplotlib figure to save
        save_path: Path to save to (or None to skip saving)
        **savefig_kwargs: Additional kwargs passed to fig.savefig()
        
    Returns:
        The resolved filepath if saved, None if save_path was None
        
    Raises:
        UnsafePathError: If path is in a protected location
    """
    if save_path is None:
        return None
    
    # Validate and prepare path (auto-increments if exists)
    filepath = safe_write_path(save_path, create_dirs=True)
    
    # Save with sensible defaults
    defaults = {"bbox_inches": "tight"}
    defaults.update(savefig_kwargs)
    fig.savefig(filepath, **defaults)
    
    return filepath


# Type alias for file types
FileType = str  # "csv", "model", or "figure"


def safe_save(
    data,
    filepath: str,
    file_type: FileType,
    **kwargs,
) -> dict:
    """
    Unified method to safely save any supported file type.
    
    Never overwrites existing files - automatically adds _1, _2, etc. suffix.
    Validates path safety (no system directories, no source code files).
    
    Args:
        data: The data to save:
            - "csv": pandas DataFrame
            - "model": any joblib-serializable object (sklearn model, etc.)
            - "figure": matplotlib Figure
        filepath: Desired output path
        file_type: One of "csv", "model", or "figure"
        **kwargs: Additional arguments for the underlying save:
            - csv: index (bool, default False), plus any df.to_csv() args
            - model: compress (int, default 0), plus any joblib.dump() args
            - figure: dpi, bbox_inches, format, etc. for fig.savefig()
    
    Returns:
        Dict with:
            - filepath: str - Actual path where file was saved
            - original_filepath: str - Originally requested path
            - was_renamed: bool - True if filename was changed to avoid overwrite
            - file_type: str - The type that was saved
    
    Raises:
        UnsafePathError: If path is in a protected location or has protected extension
        ValueError: If file_type is not recognized
        TypeError: If data type doesn't match file_type
    
    Examples:
        >>> # Save a DataFrame to CSV
        >>> result = safe_save(df, "output.csv", "csv")
        >>> result["filepath"]
        "output.csv"  # or "output_1.csv" if original existed
        
        >>> # Save a trained model
        >>> result = safe_save(model, "model.joblib", "model", compress=3)
        
        >>> # Save a matplotlib figure
        >>> result = safe_save(fig, "plot.png", "figure", dpi=300)
    """
    import pandas as pd
    
    original_filepath = filepath
    
    # Validate and get safe path (auto-increments if exists)
    safe_path = safe_write_path(filepath, create_dirs=True)
    was_renamed = safe_path != os.path.abspath(os.path.expanduser(original_filepath))
    
    # Save based on file type
    if file_type == "csv":
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame for 'csv', got {type(data).__name__}")
        index = kwargs.pop("index", False)
        data.to_csv(safe_path, index=index, **kwargs)
        
    elif file_type == "model":
        import joblib
        compress = kwargs.pop("compress", 0)
        joblib.dump(data, safe_path, compress=compress, **kwargs)
        
    elif file_type == "figure":
        # Expect matplotlib Figure
        defaults = {"bbox_inches": "tight"}
        defaults.update(kwargs)
        data.savefig(safe_path, **defaults)
        
    else:
        raise ValueError(
            f"Unknown file_type: '{file_type}'. Must be 'csv', 'model', or 'figure'"
        )
    
    return {
        "filepath": safe_path,
        "original_filepath": original_filepath,
        "was_renamed": was_renamed,
        "file_type": file_type,
    }
