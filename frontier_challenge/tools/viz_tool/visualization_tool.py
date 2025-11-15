"""
Financial Visualization Tool

This tool enables automatic generation of financial visualizations from DataFrames
and natural language queries.

Key Features:
- Automatic data summarization
- Intelligent visualization proposals
- Code generation with error recovery
- Support for multiple visualization libraries
- Financial domain expertise
"""

import logging
from io import BytesIO
from typing import Literal

import pandas as pd

from .graph import get_visualization_graph
from .models import VisualizationResult

logger = logging.getLogger(__name__)


class FinancialVisualizationTool:
    """
    Tool for generating financial visualizations from DataFrames and queries.

    This tool uses a multi-stage LangGraph workflow:
    1. Summarize dataset structure and statistics
    2. Propose relevant visualizations
    3. Generate Python code for each visualization
    4. Execute code and save results
    5. Retry on errors
    """

    def __init__(
        self,
        library: Literal["matplotlib", "seaborn", "plotly"] = "matplotlib",
        image_format: str = "png",
        output_dir: str = "/tmp",
        language: str = "portuguese",
    ):
        """
        Initialize the Financial Visualization Tool.

        Parameters
        ----------
        library : str
            Default visualization library ('matplotlib', 'seaborn', 'plotly')
        image_format : str
            Output image format ('png', 'svg', 'html')
        output_dir : str
            Directory to save generated visualizations
        language : str
            Language for labels and text ('english', 'portuguese')
        """
        self.library = library
        self.image_format = image_format
        self.output_dir = output_dir
        self.language = language

        logger.info(
            f"Initialized FinancialVisualizationTool: "
            f"library={library}, format={image_format}, output={output_dir}"
        )

    async def create_visualization(
        self,
        data: pd.DataFrame,
        query: str = "",
        number_visualizations: int = 1,
        library: str | None = None,
        image_format: str | None = None,
        output_dir: str | None = None,
    ) -> list[VisualizationResult]:
        """
        Create visualizations from a DataFrame and query.

        Parameters
        ----------
        data : pd.DataFrame
            Input data to visualize
        query : str
            Natural language query describing what to visualize
        number_visualizations : int
            Number of visualizations to generate
        library : str, optional
            Override default visualization library
        image_format : str, optional
            Override default image format
        output_dir : str, optional
            Override default output directory

        Returns
        -------
        list[VisualizationResult]
            List of generated visualizations with metadata

        Examples
        --------
        >>> tool = FinancialVisualizationTool()
        >>> results = await tool.create_visualization(
        ...     data=fund_df,
        ...     query="Compare fund returns over time",
        ...     number_visualizations=1
        ... )
        >>> print(results[0].image_path)
        '/tmp/abc123xyz.png'
        """
        library = library or self.library
        image_format = image_format or self.image_format
        output_dir = output_dir or self.output_dir

        logger.info(
            f"Creating {number_visualizations} visualization(s) for query: '{query}'"
        )
        logger.info(f"Data shape: {data.shape}, library: {library}")

        # Serialize DataFrame
        buffer = BytesIO()
        data.to_pickle(buffer)

        # Get the workflow graph
        app = get_visualization_graph(
            number_visualizations=number_visualizations,
            library=library,
            language=self.language,
            image_format=image_format,
            output_dir=output_dir,
        )

        # Execute the workflow
        result = await app.ainvoke({
            "data_frame_data": buffer.getvalue(),
            "visualization_prompt": query,
        })

        visualizations = result.get("visualizations", [])

        logger.info(f"Successfully generated {len(visualizations)} visualization(s)")

        return visualizations
