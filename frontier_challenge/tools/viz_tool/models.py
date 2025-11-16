"""Data models for the Financial Visualization Tool."""

from io import BytesIO
from typing import Annotated, Sequence

import pandas as pd
from langgraph.graph.message import AnyMessage, add_messages
from pydantic import BaseModel, Field, create_model


class VisualizationResult(BaseModel):
    """Result of a visualization generation."""

    image_path: str = Field(description="Path to the generated visualization image")
    python_code: str = Field(description="Python code used to generate the visualization")
    description: str = Field(description="Description of what the visualization shows")
    visualization_type: str = Field(description="Type of visualization (e.g., 'line chart', 'bar chart')")


class VisualizationDescription(BaseModel):
    """Description of a proposed visualization."""

    index: int
    question: str = Field(description="Financial question to answer with the visualization")
    short_question: str = Field(description="Short version of the question (5-7 words)")
    rationale: str = Field(
        description="Rationale for the proper visualization approach for this financial data"
    )
    visualization: str = Field(description="Recommended visualization type for this goal")


class VisualizationDescriptionList(BaseModel):
    """List of proposed visualizations."""

    visualizations: list[VisualizationDescription]


class CustomBaseModel(BaseModel):
    """Base model with arbitrary types allowed."""

    class Config:
        arbitrary_types_allowed = True


def create_viz_state_model(number_visualizations: int) -> type[BaseModel]:
    """
    Create a dynamic state model for the visualization workflow.

    Parameters
    ----------
    number_visualizations : int
        Number of visualizations to generate

    Returns
    -------
    type[BaseModel]
        Dynamically created state model
    """
    fields = {}

    # Branch messages fields for each visualization
    for j in range(number_visualizations):
        field_name = f"branch_{j+1}_messages"
        field_type = Annotated[Sequence[AnyMessage], add_messages] | None
        fields[field_name] = (field_type, None)

    # Visualization proposal fields
    for j in range(number_visualizations):
        field_name = f"visualization_proposal_{j+1}"
        field_type = str | None
        fields[field_name] = (field_type, None)

    # Static fields
    static_fields = {
        "visualization_prompt": (str | None, None),
        "visualization_context": (str | None, None),
        "data_frame_data": (bytes | None, None),
        "data_summary": (str | None, None),
        "visualizations": (list[VisualizationResult] | None, None),
    }
    fields.update(static_fields)

    # Create the model class dynamically
    model = create_model(
        "FinancialVizStateModel",
        __base__=CustomBaseModel,
        **fields,
    )

    # Add the data_frame property
    def data_frame(self):
        if self.data_frame_data is not None:
            return pd.read_pickle(BytesIO(self.data_frame_data))
        else:
            return None

    setattr(model, "data_frame", property(data_frame))

    return model
