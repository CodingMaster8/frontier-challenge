"""Stages for the Financial Visualization workflow."""

import json
import logging

import pandas as pd
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from pydantic import Field

from .models import VisualizationDescription, VisualizationDescriptionList, CustomBaseModel
from .prompts import SUMMARIZATION_PROMPT, PROPOSAL_PROMPT, GENERATION_PROMPT
from .utils import get_column_properties, get_chart_template

logger = logging.getLogger(__name__)

class DatasetColumn(CustomBaseModel):
    column_name: str
    description: str
    data_type: str

class DatasetInfo(CustomBaseModel):
    overall_description: str
    columns: list[DatasetColumn]
    key_insights: list[str] = Field(description="Key insights for visualization")

def summarize_dataset(
    data: pd.DataFrame,
    context: str = "",
    n_samples: int = 3,
    llm=None,
) -> str:
    """
    Summarize the DataFrame for visualization purposes.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame
    context : str
        Additional context about the data
    n_samples : int
        Number of sample values per column
    llm : LLM
        Language model to use

    Returns
    -------
    str
        JSON string with dataset summary
    """
    if llm is None:
        raise ValueError("LLM must be provided")

    # Get column properties
    data_properties = get_column_properties(data, n_samples)

    parser = PydanticOutputParser(pydantic_object=DatasetInfo)

    prompt = PromptTemplate(
        template=SUMMARIZATION_PROMPT,
        input_variables=["schema", "context_str"],
        partial_variables={
            "format_instructions": parser.get_format_instructions(),
            "nrows": len(data),
        },
    )

    chain = prompt | llm | parser

    dataset_info = chain.invoke({
        "schema": str(data_properties),
        "context_str": context
    }).dict()

    dataset_info["num_rows"] = len(data)
    for column in dataset_info["columns"]:
        if column["column_name"] in data_properties:
            column["properties"] = data_properties[column["column_name"]]

    return str(dataset_info)


def generate_visualization_proposals(
    data_summary: str,
    context: str = "",
    n: int = 1,
    llm=None,
) -> VisualizationDescriptionList:
    """
    Generate visualization proposals based on data summary.

    Parameters
    ----------
    data_summary : str
        Summary of the dataset
    context : str
        User context or query
    n : int
        Number of visualizations to propose
    llm : LLM
        Language model to use

    Returns
    -------
    VisualizationDescriptionList
        List of proposed visualizations
    """
    if llm is None:
        raise ValueError("LLM must be provided")

    context_str = f"<context>{context}</context>" if context else ""

    parser = PydanticOutputParser(pydantic_object=VisualizationDescriptionList)
    prompt = PromptTemplate(
        template=PROPOSAL_PROMPT,
        input_variables=["data_summary", "context_str"],
        partial_variables={
            "FORMAT_INSTRUCTIONS": parser.get_format_instructions(),
            "n_visualizations": n,
        },
    )

    chain = prompt | llm | parser

    return chain.invoke({
        "data_summary": data_summary,
        "context_str": context
    })


def generate_visualization_code(
    data_summary: str,
    visualization_description: VisualizationDescription,
    context: str = "",
    previous_messages: list = None,
    library: str = "matplotlib",
    language: str = "english",
    llm=None,
) -> str:
    """
    Generate Python code for a visualization.

    Parameters
    ----------
    data_summary : str
        Summary of the dataset
    visualization_description : VisualizationDescription
        Description of the visualization to generate
    context : str
        Additional context
    previous_messages : list
        Previous conversation messages (for error recovery)
    library : str
        Visualization library to use
    language : str
        Language for labels and text
    llm : LLM
        Language model to use

    Returns
    -------
    str
        Python code for the visualization
    """
    if llm is None:
        raise ValueError("LLM must be provided")

    library_template, library_instructions = get_chart_template(library)

    context_str = f"<context>{context}</context>" if context else ""

    system_prompt = GENERATION_PROMPT.format(
        visualization=visualization_description.visualization,
        question=visualization_description.question,
        short_question=visualization_description.short_question,
        language=language,
        library=library,
        library_instructions=library_instructions,
        data_summary=data_summary,
        context_str=context_str,
        library_template=library_template.strip(),
    )

    input_messages = [HumanMessage(content=system_prompt)]

    if previous_messages:
        input_messages.extend(previous_messages)

    message = llm.invoke(input_messages)

    return message.content
