"""
File Safety Utilities.

Provides safe file operations that prevent accidental overwrites
and validate paths to avoid dangerous operations.

On a laptop a caller's path is the user's own choice, and that is the default.
A server confines each session with a ``FilePolicy`` on its ``DataFrameState``
(or the STATS_COMPASS_WRITE_ROOT / STATS_COMPASS_READ_ROOTS environment
variables for every state): writes land inside the write root (a path outside
it is moved in by base name) with an extension that fits the file, and reads
must resolve inside a read root. Containment is checked on the real path, so a symlink cannot lead out
(security scan F6, F8, 8 Oct 2026). The system-folder denylist below stays as a
second line for unconfined use; the root is the control.
"""

import os
from dataclasses import dataclass
from pathlib import Path


class UnsafePathError(ValueError):
    """Raised when a path is deemed unsafe (e.g., system directories, sensitive locations).

    A ValueError, so callers that treat bad input as ValueError catch it too.
    """


@dataclass(frozen=True)
class FilePolicy:
    """Where a session's tools may write and read. None means unconfined.

    ``write_root``: every write lands inside it. ``read_roots``: every
    read and listing must resolve inside one of these; a relative path is taken
    relative to the first.
    """

    write_root: Path | None = None
    read_roots: tuple[Path, ...] | None = None

    @classmethod
    def from_env(cls) -> "FilePolicy":
        """STATS_COMPASS_WRITE_ROOT, and STATS_COMPASS_READ_ROOTS separated by os.pathsep."""
        write = os.getenv("STATS_COMPASS_WRITE_ROOT") or None
        reads = os.getenv("STATS_COMPASS_READ_ROOTS") or None
        return cls(
            write_root=Path(write) if write else None,
            read_roots=tuple(Path(p) for p in reads.split(os.pathsep) if p) if reads else None,
        )


# What each kind of file may be called when written under a root.
ALLOWED_EXTENSIONS: dict[str, set[str]] = {
    "csv": {".csv", ".tsv", ".txt"},
    "model": {".joblib", ".pkl", ".pickle"},
    "figure": {".png", ".svg", ".pdf", ".jpg", ".jpeg"},
}


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
    # Expand and resolve path. Both the plain and the real path are checked, against
    # both forms of each forbidden folder: a symlink is judged by where it leads, and
    # on macOS /etc itself is a link to /private/etc.
    expanded = os.path.expanduser(filepath)
    resolved = os.path.abspath(expanded)
    path = Path(resolved)
    candidates = {resolved, os.path.realpath(expanded)}

    # Check for forbidden parent directories
    for forbidden in FORBIDDEN_PATHS:
        for forbidden_form in {forbidden, os.path.realpath(forbidden)}:
            forbidden_path = Path(forbidden_form)
            if not forbidden_path.exists():
                continue
            for candidate in candidates:
                try:
                    # Check if the path is under a forbidden directory
                    if candidate.startswith(str(forbidden_path) + os.sep):
                        # Allow if it's deep enough (user subdirectory)
                        relative = Path(candidate).relative_to(forbidden_path)
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
    *,
    root: str | os.PathLike | None = None,
    file_type: str | None = None,
) -> str:
    """
    Validate and prepare a path for safe writing.
    
    Never overwrites existing files - automatically adds numeric suffix
    (e.g., output_1.csv, output_2.csv) if file exists.
    
    Args:
        filepath: The target path
        create_dirs: If True, create parent directories if they don't exist
        root: If given, the file lands inside this folder: a path inside it
            keeps its folders, anything else is moved in by base name. Its
            extension must fit ``file_type``.
        file_type: "csv", "model" or "figure"; checked against
            ALLOWED_EXTENSIONS when ``root`` is given.
        
    Returns:
        The resolved absolute path (may have _N suffix if original existed)
        
    Raises:
        UnsafePathError: If the path is in a forbidden location or has a protected extension
    """
    if root is not None:
        return _write_path_in_root(filepath, root, file_type)

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


def _write_path_in_root(filepath: str, root: str | os.PathLike, file_type: str | None) -> str:
    """Where ``filepath`` may be written under ``root``, checked on the real path.

    A relative path is taken relative to the root. If it then lies inside the
    root, its folders are kept, so a server's own layout (``data/``,
    ``models/``) survives. Anything that would land outside, through an
    absolute path, ``..`` or a symlinked folder, is written to the root under
    its base name instead.
    """
    name = Path(os.path.expanduser(str(filepath))).name
    if name in ("", ".", ".."):
        raise UnsafePathError(f"'{filepath}' does not name a file.")
    suffix = Path(name).suffix.lower()
    if file_type is not None:
        allowed = ALLOWED_EXTENSIONS.get(file_type)
        if allowed is None:
            raise ValueError(f"Unknown file_type: '{file_type}'")
        if suffix not in allowed:
            raise UnsafePathError(
                f"A {file_type} file must end in one of {', '.join(sorted(allowed))}; got '{name}'."
            )
    root_real = os.path.realpath(os.path.expanduser(str(root)))
    os.makedirs(root_real, exist_ok=True)
    expanded = os.path.expanduser(str(filepath))
    requested = expanded if os.path.isabs(expanded) else os.path.join(root_real, expanded)
    requested_real = os.path.realpath(requested)
    target = requested_real if _inside(requested_real, root_real) else os.path.join(root_real, name)
    is_safe, error = is_path_safe(target)
    if not is_safe:
        raise UnsafePathError(error)
    os.makedirs(os.path.dirname(target), exist_ok=True)
    candidate = get_unique_filepath(target)
    # A symlink planted in the root, even a dangling one, would lead the write out.
    while os.path.islink(candidate):
        stem, ext = os.path.splitext(candidate)
        candidate = get_unique_filepath(f"{stem}_1{ext}")
    if not _inside(os.path.realpath(candidate), root_real):
        raise UnsafePathError(f"'{filepath}' resolves outside the output folder.")
    return candidate


def check_read_path(
    filepath: str,
    roots: tuple[str | os.PathLike, ...] | list | None,
) -> str:
    """The path to read, or UnsafePathError if a policy confines reads and it leaves the roots.

    With no roots, the path is only ``~``-expanded, as before. With roots, a
    relative path is taken relative to the first root, and the real path (after
    symlinks and ``..``) must lie inside one of them.
    """
    expanded = os.path.expanduser(str(filepath))
    if not roots:
        return expanded
    reals = [os.path.realpath(os.path.expanduser(str(r))) for r in roots]
    if not os.path.isabs(expanded):
        expanded = os.path.join(reals[0], expanded)
    real = os.path.realpath(expanded)
    if not any(_inside(real, r) for r in reals):
        raise UnsafePathError(f"'{filepath}' is outside the folders this session may read.")
    return real


def _inside(path: str, root: str) -> bool:
    try:
        return os.path.commonpath([path, root]) == root
    except ValueError:  # different drives on Windows
        return False


def safe_save_figure(
    fig,
    save_path: str | None,
    *,
    root: str | os.PathLike | None = None,
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
    filepath = safe_write_path(save_path, create_dirs=True, root=root, file_type="figure")

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
    *,
    root: str | os.PathLike | None = None,
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
        root: If given, the file lands inside this folder (a path outside it
            is moved in by base name), with an extension that fits
            ``file_type`` (a session's FilePolicy.write_root)
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
    
    from .spreadsheet_safety import sanitize_dataframe

    original_filepath = filepath

    if file_type not in ("csv", "model", "figure"):
        raise ValueError(
            f"Unknown file_type: '{file_type}'. Must be 'csv', 'model', or 'figure'"
        )

    # Validate and get safe path (auto-increments if exists)
    safe_path = safe_write_path(filepath, create_dirs=True, root=root, file_type=file_type)
    if root is None:
        was_renamed = safe_path != os.path.abspath(os.path.expanduser(original_filepath))
    else:
        # Renamed means the name changed (a _1 suffix), not that it was moved into the root.
        was_renamed = os.path.basename(safe_path) != Path(os.path.expanduser(original_filepath)).name

    # Save based on file type
    if file_type == "csv":
        if not isinstance(data, pd.DataFrame):
            raise TypeError(f"Expected DataFrame for 'csv', got {type(data).__name__}")
        index = kwargs.pop("index", False)
        # Sanitize to prevent CSV injection attacks (formula injection)
        safe_data = sanitize_dataframe(data)
        safe_data.to_csv(safe_path, index=index, **kwargs)

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
