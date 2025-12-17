import os
import joblib
from pydantic import Field

from stats_compass_core.base import StrictToolInput
from stats_compass_core.registry import registry
from stats_compass_core.state import DataFrameState


class SaveModelInput(StrictToolInput):
    """Input for saving a trained model to a file."""

    model_id: str = Field(..., description="ID of the model to save")
    filepath: str = Field(..., description="Path where the model file will be saved (e.g., model.joblib)")


@registry.register(
    category="ml",
    name="save_model",
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
        A dictionary with a success message.
    """
    # Get the model from state
    model = state.get_model(input_data.model_id)
    if model is None:
        raise ValueError(f"Model '{input_data.model_id}' not found.")

    # Resolve path (handle ~)
    filepath = os.path.expanduser(input_data.filepath)
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)

    # Save model
    joblib.dump(model, filepath)

    return {
        "message": f"Model '{input_data.model_id}' saved to '{filepath}'",
        "filepath": filepath,
    }
