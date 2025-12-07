"""
DataFrame state manager for MCP-compatible tool execution.

This module provides the central state management for stats-compass-core,
allowing tools to operate on DataFrames stored server-side without passing
raw data across the MCP protocol boundary.

Key Features:
- Multiple DataFrame support (like a data scientist's workspace)
- Memory limit enforcement to prevent crashes
- Model storage with descriptive naming
- Operation history/lineage tracking
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

import pandas as pd
from pydantic import BaseModel


class DataFrameInfo(BaseModel):
    """Metadata about a stored DataFrame."""

    name: str
    shape: tuple[int, int]
    columns: list[str]
    dtypes: dict[str, str]
    memory_mb: float
    created_at: str
    source_operation: str


class ModelInfo(BaseModel):
    """Metadata about a stored model."""

    model_id: str
    model_type: str
    target_column: str
    feature_columns: list[str]
    created_at: str
    source_dataframe: str


class HistoryEntry(BaseModel):
    """Record of an operation performed on state."""

    timestamp: str
    operation: str
    dataframe_name: str
    details: dict[str, Any] = {}


class DataFrameState:
    """
    Server-side state manager for MCP sessions.

    Manages multiple DataFrames, trained models, and operation history.
    Enforces memory limits to prevent crashes.

    Example:
        state = DataFrameState(memory_limit_mb=500)
        state.set_dataframe(df, name="sales_data", operation="load_csv")
        state.set_dataframe(df2, name="aggregated", operation="groupby")

        # Tools access state
        df = state.get_dataframe("sales_data")
        state.store_model(model, "linear_regression", "price", features, "sales_data")
    """

    def __init__(self, memory_limit_mb: float = 500.0) -> None:
        """
        Initialize state manager.

        Args:
            memory_limit_mb: Maximum total memory for all DataFrames (default 500MB)
        """
        self._dataframes: dict[str, pd.DataFrame] = {}
        self._dataframe_metadata: dict[str, DataFrameInfo] = {}
        self._models: dict[str, Any] = {}
        self._model_metadata: dict[str, ModelInfo] = {}
        self._history: list[HistoryEntry] = []
        self._memory_limit_mb = memory_limit_mb
        self._active_dataframe: str | None = None  # Currently selected DataFrame

    # =========================================================================
    # DataFrame Management
    # =========================================================================

    def set_dataframe(
        self,
        df: pd.DataFrame,
        name: str,
        operation: str,
        set_active: bool = True,
    ) -> str:
        """
        Store a DataFrame in state.

        Args:
            df: DataFrame to store
            name: Unique name for this DataFrame
            operation: Operation that created this DataFrame (for lineage)
            set_active: Whether to set this as the active DataFrame

        Returns:
            The name of the stored DataFrame (for consistency)

        Raises:
            MemoryError: If adding this DataFrame would exceed memory limit
            ValueError: If name is empty
        """
        if not name:
            raise ValueError("DataFrame name cannot be empty")

        # Calculate memory usage
        memory_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)

        # Check memory limit (excluding current df if overwriting)
        current_usage = self.get_total_memory_mb()
        if name in self._dataframes:
            current_usage -= self._dataframe_metadata[name].memory_mb

        if current_usage + memory_mb > self._memory_limit_mb:
            raise MemoryError(
                f"Cannot store DataFrame '{name}' ({memory_mb:.1f}MB). "
                f"Would exceed memory limit of {self._memory_limit_mb:.1f}MB. "
                f"Current usage: {current_usage:.1f}MB. "
                f"Consider removing unused DataFrames with remove_dataframe()."
            )

        # Store DataFrame
        self._dataframes[name] = df

        # Store metadata
        info = DataFrameInfo(
            name=name,
            shape=df.shape,
            columns=df.columns.tolist(),
            dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
            memory_mb=memory_mb,
            created_at=datetime.now().isoformat(),
            source_operation=operation,
        )
        self._dataframe_metadata[name] = info

        # Record in history
        self._add_history(operation, name, {"shape": df.shape, "memory_mb": memory_mb})

        # Set as active if requested
        if set_active:
            self._active_dataframe = name

        return name

    def get_dataframe(self, name: str | None = None) -> pd.DataFrame:
        """
        Get a DataFrame from state.

        Args:
            name: Name of DataFrame to retrieve. If None, returns active DataFrame.

        Returns:
            The requested DataFrame

        Raises:
            ValueError: If DataFrame not found or no active DataFrame
        """
        if name is None:
            name = self._active_dataframe
            if name is None:
                raise ValueError("No active DataFrame. Load data first or specify name.")

        if name not in self._dataframes:
            available = list(self._dataframes.keys())
            raise ValueError(
                f"DataFrame '{name}' not found. Available: {available}"
            )

        return self._dataframes[name]

    def get_dataframe_info(self, name: str | None = None) -> DataFrameInfo:
        """Get metadata about a DataFrame."""
        if name is None:
            name = self._active_dataframe
            if name is None:
                raise ValueError("No active DataFrame.")

        if name not in self._dataframe_metadata:
            raise ValueError(f"DataFrame '{name}' not found.")

        return self._dataframe_metadata[name]

    def list_dataframes(self) -> list[DataFrameInfo]:
        """List all stored DataFrames with their metadata."""
        return list(self._dataframe_metadata.values())

    def remove_dataframe(self, name: str) -> None:
        """Remove a DataFrame from state to free memory."""
        if name not in self._dataframes:
            raise ValueError(f"DataFrame '{name}' not found.")

        del self._dataframes[name]
        del self._dataframe_metadata[name]
        self._add_history("remove_dataframe", name, {})

        # Clear active if it was removed
        if self._active_dataframe == name:
            self._active_dataframe = None

    def set_active_dataframe(self, name: str) -> None:
        """Set the active DataFrame for subsequent operations."""
        if name not in self._dataframes:
            raise ValueError(f"DataFrame '{name}' not found.")
        self._active_dataframe = name

    def get_active_dataframe_name(self) -> str | None:
        """Get the name of the currently active DataFrame."""
        return self._active_dataframe

    def has_dataframe(self, name: str | None = None) -> bool:
        """Check if a DataFrame exists in state."""
        if name is None:
            return self._active_dataframe is not None
        return name in self._dataframes

    # =========================================================================
    # Model Management
    # =========================================================================

    def store_model(
        self,
        model: Any,
        model_type: str,
        target_column: str,
        feature_columns: list[str],
        source_dataframe: str,
        custom_name: str | None = None,
    ) -> str:
        """
        Store a trained model in state.

        Args:
            model: The trained model object (sklearn, statsmodels, etc.)
            model_type: Type of model ("linear_regression", "random_forest", etc.)
            target_column: Name of target column
            feature_columns: Names of feature columns
            source_dataframe: Name of DataFrame used for training
            custom_name: Optional custom name for the model

        Returns:
            model_id: Descriptive identifier for the model
        """
        # Generate descriptive model ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if custom_name:
            model_id = f"{custom_name}_{timestamp}"
        else:
            model_id = f"{model_type}_{target_column}_{timestamp}"

        # Ensure uniqueness
        if model_id in self._models:
            model_id = f"{model_id}_{uuid.uuid4().hex[:6]}"

        # Store model
        self._models[model_id] = model

        # Store metadata
        info = ModelInfo(
            model_id=model_id,
            model_type=model_type,
            target_column=target_column,
            feature_columns=feature_columns,
            created_at=datetime.now().isoformat(),
            source_dataframe=source_dataframe,
        )
        self._model_metadata[model_id] = info

        # Record in history
        self._add_history(
            f"train_{model_type}",
            source_dataframe,
            {"model_id": model_id, "target": target_column},
        )

        return model_id

    def get_model(self, model_id: str) -> Any:
        """Get a trained model by ID."""
        # Return None if model not found to make callers' checks simpler
        return self._models.get(model_id)

    def get_model_info(self, model_id: str) -> ModelInfo:
        """Get metadata about a trained model."""
        if model_id not in self._model_metadata:
            raise ValueError(f"Model '{model_id}' not found.")
        return self._model_metadata[model_id]

    def list_models(self) -> list[ModelInfo]:
        """List all stored models with their metadata."""
        return list(self._model_metadata.values())

    def remove_model(self, model_id: str) -> None:
        """Remove a model from state."""
        if model_id not in self._models:
            raise ValueError(f"Model '{model_id}' not found.")
        del self._models[model_id]
        del self._model_metadata[model_id]

    # =========================================================================
    # Memory Management
    # =========================================================================

    def get_total_memory_mb(self) -> float:
        """Get total memory used by all DataFrames."""
        return sum(info.memory_mb for info in self._dataframe_metadata.values())

    def get_memory_status(self) -> dict[str, float]:
        """Get memory usage status."""
        used = self.get_total_memory_mb()
        return {
            "used_mb": used,
            "limit_mb": self._memory_limit_mb,
            "available_mb": self._memory_limit_mb - used,
            "usage_percent": (used / self._memory_limit_mb) * 100,
        }

    # =========================================================================
    # History / Lineage
    # =========================================================================

    def get_history(self) -> list[HistoryEntry]:
        """Get full operation history."""
        return self._history.copy()

    def _add_history(
        self, operation: str, dataframe_name: str, details: dict[str, Any]
    ) -> None:
        """Add an entry to operation history."""
        entry = HistoryEntry(
            timestamp=datetime.now().isoformat(),
            operation=operation,
            dataframe_name=dataframe_name,
            details=details,
        )
        self._history.append(entry)

    # =========================================================================
    # Serialization (for MCP)
    # =========================================================================

    def get_state_summary(self) -> dict[str, Any]:
        """
        Get a JSON-serializable summary of current state.

        Useful for MCP clients to understand what data is available.
        """
        return {
            "active_dataframe": self._active_dataframe,
            "dataframes": [info.model_dump() for info in self._dataframe_metadata.values()],
            "models": [info.model_dump() for info in self._model_metadata.values()],
            "memory": self.get_memory_status(),
            "history_length": len(self._history),
        }
