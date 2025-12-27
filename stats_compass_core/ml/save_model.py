from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState
from stats_compass_core.utils import safe_save, UnsafePathError


class SaveModelInput(StrictToolInput):
    """Input for saving a trained model to a file."""

    model_id: str = Field(..., description="ID of the model to save")
    filepath: str = Field(..., description="Path where the model file will be saved (e.g., model.joblib)")


@registry.register(
    category="ml",
    name="save_model",
    tier="util",
    input_schema=SaveModelInput,
    description="Save a trained model to a file using joblib.",
)
def save_model(state: DataFrameState, input_data: SaveModelInput) -> dict[str, str]:
    """
    Save a trained model to a file.

    Args:
        state: The DataFrameState manager.
        input_data: The input parameters.

    Returns:
        A dictionary with a success message and actual filepath used.
        
    Raises:
        ValueError: If model not found
        UnsafePathError: If path is in a protected location or has protected extension
    """
    # Get the model from state
    model = state.get_model(input_data.model_id)
    if model is None:
        raise ValueError(f"Model '{input_data.model_id}' not found.")

    # Use unified safe_save
    result = safe_save(model, input_data.filepath, "model")

    return {
        "message": f"Model '{input_data.model_id}' saved to '{result['filepath']}'",
        "filepath": result["filepath"],
        "was_renamed": str(result["was_renamed"]),
    }
