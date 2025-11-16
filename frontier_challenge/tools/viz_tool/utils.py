"""Utility functions for the Financial Visualization Tool."""

import logging
import os
import random
import string
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def generate_random_name(length: int = 12) -> str:
    """
    Generate a random string for file naming.

    Parameters
    ----------
    length : int
        Length of the random string

    Returns
    -------
    str
        Random alphanumeric string
    """
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))


def get_column_properties(df: pd.DataFrame, n_samples: int = 3) -> dict[str, Any]:
    """
    Extract properties for each column in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    n_samples : int
        Number of sample values to include

    Returns
    -------
    dict
        Dictionary with column properties
    """
    properties = {}

    for col in df.columns:
        col_data = df[col]
        prop = {
            "dtype": str(col_data.dtype),
            "missing_count": int(col_data.isna().sum()),
            "missing_ratio": float(col_data.isna().sum() / len(col_data)),
            "unique_count": int(col_data.nunique()),
        }

        # Add samples (non-null values)
        non_null = col_data.dropna()
        if len(non_null) > 0:
            samples = non_null.sample(min(n_samples, len(non_null))).tolist()
            prop["samples"] = [str(s) for s in samples]
        else:
            prop["samples"] = []

        # Add numeric statistics if applicable
        if pd.api.types.is_numeric_dtype(col_data):
            prop["min"] = float(col_data.min()) if not col_data.isna().all() else None
            prop["max"] = float(col_data.max()) if not col_data.isna().all() else None
            prop["mean"] = float(col_data.mean()) if not col_data.isna().all() else None
            prop["median"] = float(col_data.median()) if not col_data.isna().all() else None
            prop["std"] = float(col_data.std()) if not col_data.isna().all() else None

            # Calculate additional quality metrics
            if prop["max"] and prop["min"]:
                value_range = prop["max"] - prop["min"]
                if value_range > 0 and prop["std"]:
                    prop["std_ratio"] = prop["std"] / value_range

        properties[col] = prop

    return properties


def preprocess_code(content: str) -> str:
    """
    Extract Python code from LLM response.

    Parameters
    ----------
    content : str
        LLM response content

    Returns
    -------
    str
        Extracted Python code
    """
    # Look for code blocks
    if "```python" in content:
        start = content.find("```python") + len("```python")
        end = content.find("```", start)
        return content[start:end].strip()
    elif "```" in content:
        start = content.find("```") + len("```")
        end = content.find("```", start)
        return content[start:end].strip()

    # If no code blocks, return as is
    return content.strip()


def unsafe_execute_code(code: str, data: pd.DataFrame) -> Any:
    """
    Execute Python code with the DataFrame in scope.

    WARNING: This executes arbitrary code. Use only in controlled environments.

    Parameters
    ----------
    code : str
        Python code to execute
    data : pd.DataFrame
        DataFrame to make available to the code

    Returns
    -------
    Any
        The 'chart' variable from executed code
    """
    local_vars = {"data": data, "pd": pd}

    # Execute the code
    exec(code, {"__builtins__": __builtins__}, local_vars)

    # Return the chart object
    if "chart" in local_vars:
        return local_vars["chart"]
    else:
        raise ValueError("Code did not produce a 'chart' variable")


def get_chart_template(library: str) -> tuple[str, str]:
    """
    Get the chart template and instructions for a given library.

    Parameters
    ----------
    library : str
        Visualization library name

    Returns
    -------
    tuple[str, str]
        (template_code, library_instructions)
    """
    if library == "matplotlib":
        template = """
            <imports>
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            </imports>

            # The 'data' DataFrame is already loaded
            <stub>
            # Your visualization code here
            # Create figure and plot
            # fig, ax = plt.subplots(figsize=(12, 6))
            # ... your plotting code ...
            </stub>

            chart = fig
        """
        from .prompts import MATPLOTLIB_INSTRUCTIONS

        return template, MATPLOTLIB_INSTRUCTIONS

    elif library == "seaborn":
        template = """
            <imports>
            import seaborn as sns
            import matplotlib.pyplot as plt
            import pandas as pd
            import numpy as np
            </imports>

            # The 'data' DataFrame is already loaded
            sns.set_theme(style='whitegrid')

            <stub>
            # Your visualization code here
            # fig, ax = plt.subplots(figsize=(12, 6))
            # ... your seaborn plotting code ...
            </stub>

            chart = fig
        """
        from .prompts import SEABORN_INSTRUCTIONS

        return template, SEABORN_INSTRUCTIONS

    elif library == "plotly":
        template = """
            <imports>
            import plotly.graph_objects as go
            import plotly.express as px
            import pandas as pd
            import numpy as np
            </imports>

            # The 'data' DataFrame is already loaded
            <stub>
            # Your visualization code here
            # fig = px.line(data, x='...', y='...')
            # or
            # fig = go.Figure()
            # fig.add_trace(...)
            # fig.update_layout(...)
            </stub>

            chart = fig
        """
        from .prompts import PLOTLY_INSTRUCTIONS

        return template, PLOTLY_INSTRUCTIONS

    else:
        raise ValueError(f"Unsupported library: {library}")


def save_chart(chart: Any, file_path: str, library: str = "matplotlib") -> None:
    """
    Save a chart to a file.

    Parameters
    ----------
    chart : Any
        Chart object (matplotlib Figure or plotly Figure)
    file_path : str
        Path where to save the chart
    library : str
        Library used to create the chart
    """
    if library in ["matplotlib", "seaborn"]:
        chart.savefig(file_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved matplotlib chart to {file_path}")
    elif library == "plotly":
        if file_path.endswith('.html'):
            chart.write_html(file_path)
        else:
            chart.write_image(file_path)
        logger.info(f"Saved plotly chart to {file_path}")
    else:
        raise ValueError(f"Unsupported library for saving: {library}")
