"""Main graph/workflow for the Financial Visualization Tool."""

import json
import logging
import os
import time
from io import BytesIO
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

from ...settings import OPENAI_API_KEY
from .models import VisualizationDescription, VisualizationResult, create_viz_state_model
from .stages import (
    summarize_dataset,
    generate_visualization_proposals,
    generate_visualization_code,
)
from .utils import generate_random_name, preprocess_code, unsafe_execute_code, save_chart

logger = logging.getLogger(__name__)


def get_visualization_graph(
    number_visualizations: int = 1,
    library: str = "seaborn",
    language: str = "english",
    image_format: str = "png",
    output_dir: str = "/tmp",
):
    """
    LangGraph workflow for financial visualization generation.

    Parameters
    ----------
    number_visualizations : int
        Number of visualizations to generate
    library : str
        Visualization library ('matplotlib', 'seaborn', 'plotly')
    language : str
        Language for labels and text
    image_format : str
        Output image format ('png', 'svg', 'html')
    output_dir : str
        Directory to save generated visualizations

    Returns
    -------
    CompiledGraph
        Compiled LangGraph workflow
    """
    # Initialize LLMs
    llm_light = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=OPENAI_API_KEY
    ).with_retry()

    llm_main = ChatOpenAI(
        model="gpt-4o",
        temperature=0.3,
        openai_api_key=OPENAI_API_KEY
    ).with_retry()

    llm_code = ChatOpenAI(
        model="o4-mini",
        openai_api_key=OPENAI_API_KEY
    ).with_retry()

    State = create_viz_state_model(number_visualizations)

    workflow = StateGraph(State)

    # Node 1: Summarize Dataset
    async def node_summarize_dataset(state: State) -> dict:
        """Summarize the dataset for visualization."""
        start_time = time.time()
        context = state.visualization_prompt or ""
        df = state.data_frame

        logger.info(f"Summarizing dataset with {len(df)} rows and {len(df.columns)} columns")

        data_summary = summarize_dataset(
            df,
            context=context,
            llm=llm_light
        )

        elapsed = time.time() - start_time
        logger.info(f"Dataset summarization completed in {elapsed:.2f}s")

        return {"data_summary": data_summary}

    # Node 2: Propose Visualizations
    async def node_propose_visualizations(state: State) -> dict:
        """Generate visualization proposals."""
        start_time = time.time()
        logger.info(f"Proposing {number_visualizations} visualization(s)")

        proposed_visualizations = generate_visualization_proposals(
            state.data_summary,
            context=state.visualization_prompt or "",
            n=number_visualizations,
            llm=llm_main,
        ).visualizations

        return_dict = {
            f"visualization_proposal_{i+1}": x.json()
            for i, x in enumerate(proposed_visualizations)
        }

        elapsed = time.time() - start_time
        logger.info(f"✅ Generated {len(proposed_visualizations)} proposals in {elapsed:.2f}s")
        return return_dict

    # Node 3: Await completion (synchronization point)
    async def node_await_visualizations(state: State) -> dict:
        """Wait for all visualizations to complete."""
        return {"visualization_prompt": state.visualization_prompt}

    # Dynamic node generators for each visualization
    def get_node_generate(i: int):
        async def node_generate_visualization(state: State) -> dict:
            """Generate visualization code."""
            start_time = time.time()
            previous_messages = getattr(state, f"branch_{i}_messages")
            visualization_description_json = json.loads(
                getattr(state, f"visualization_proposal_{i}"),
                strict=False
            )
            visualization_description = VisualizationDescription.parse_obj(
                visualization_description_json
            )

            logger.info(f"Generating code for visualization {i}: {visualization_description.short_question}")

            code = generate_visualization_code(
                state.data_summary,
                visualization_description,
                previous_messages=previous_messages,
                context=state.visualization_context or state.visualization_prompt or "",
                library=library,
                language=language,
                llm=llm_code,
            )

            elapsed = time.time() - start_time
            logger.info(f"✅ Code generation for visualization {i} completed in {elapsed:.2f}s")

            return {f"branch_{i}_messages": [AIMessage(content=code)]}

        return node_generate_visualization

    def get_node_execute(i: int):
        async def node_execute_visualization_code(state: State) -> dict:
            """Execute visualization code and save result."""
            start_time = time.time()
            last_message = getattr(state, f"branch_{i}_messages")[-1]

            try:
                preprocess_start = time.time()
                python_code = preprocess_code(last_message.content)
                logger.info(f"Code preprocessing for visualization {i} took {time.time() - preprocess_start:.2f}s")
            except Exception as e:
                logger.error(f"❌ Failed to preprocess code for visualization {i}: {e}")
                return {
                    f"branch_{i}_messages": [
                        last_message,
                        HumanMessage(
                            content="Error: Code format is incorrect. Please fix and try again."
                        ),
                    ]
                }

            result = None
            try:
                file_name = generate_random_name() + f".{image_format}"
                file_path = os.path.join(output_dir, file_name)

                logger.info(f"🚀 Executing visualization code for {file_name}")

                # Execute the code
                exec_start = time.time()
                chart = unsafe_execute_code(python_code, state.data_frame)
                logger.info(f"Code execution took {time.time() - exec_start:.2f}s")

                # Save the chart
                save_start = time.time()
                save_chart(chart, file_path, library=library)
                logger.info(f"Chart save took {time.time() - save_start:.2f}s")

                # Get visualization description
                viz_desc_json = json.loads(
                    getattr(state, f"visualization_proposal_{i}"),
                    strict=False
                )
                viz_desc = VisualizationDescription.parse_obj(viz_desc_json)

                result = VisualizationResult(
                    python_code=python_code,
                    image_path=file_path,
                    description=viz_desc.question,
                    visualization_type=viz_desc.visualization,
                ).model_dump_json()

                elapsed = time.time() - start_time
                logger.info(f"Successfully generated visualization {i}: {file_path} (total: {elapsed:.2f}s)")

            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"Error executing visualization {i} code after {elapsed:.2f}s: {e}")
                result = f"Error: Your code yielded this error: {e}. Please consider what went wrong and fix your answer."

            return {
                f"branch_{i}_messages": [last_message, HumanMessage(content=result)]
            }

        return node_execute_visualization_code

    def get_node_accept(i: int):
        async def node_accept_visualization(state: State) -> dict:
            """Accept a visualization result as valid."""
            visualization_result_json = getattr(state, f"branch_{i}_messages")[-1].content

            return {
                "visualizations": [
                    VisualizationResult.model_validate_json(visualization_result_json)
                ]
            }

        return node_accept_visualization

    def get_error_conditional_edge(i, viz_gen_step_name, viz_accept_step_name):
        async def error_in_generation(state: State) -> Literal[viz_gen_step_name, viz_accept_step_name]:
            last_message = getattr(state, f"branch_{i}_messages")[-1].content
            if last_message.startswith("Error"):
                logger.warning(f"⚠️  Visualization {i} failed, retrying...")
                return viz_gen_step_name
            else:
                return viz_accept_step_name

        return error_in_generation

    # Build the graph
    workflow.add_node("summarize_dataset", node_summarize_dataset)
    workflow.add_edge(START, "summarize_dataset")

    workflow.add_node("propose_visualizations", node_propose_visualizations)
    workflow.add_edge("summarize_dataset", "propose_visualizations")

    workflow.add_node("await_visualizations", node_await_visualizations)

    accept_steps = []
    for i in range(number_visualizations):
        viz_gen_step_name = f"generate_visualization_{i+1}"
        viz_exec_step_name = f"execute_visualization_{i+1}"
        viz_accept_step_name = f"accept_visualization_{i+1}"

        node_gen = get_node_generate(i + 1)
        node_exec = get_node_execute(i + 1)
        node_accept = get_node_accept(i + 1)

        workflow.add_node(viz_gen_step_name, node_gen)
        workflow.add_edge("propose_visualizations", viz_gen_step_name)

        workflow.add_node(viz_exec_step_name, node_exec)
        workflow.add_edge(viz_gen_step_name, viz_exec_step_name)

        workflow.add_node(viz_accept_step_name, node_accept)
        workflow.add_conditional_edges(
            viz_exec_step_name,
            get_error_conditional_edge(i + 1, viz_gen_step_name, viz_accept_step_name),
        )

        accept_steps.append(viz_accept_step_name)

    workflow.add_edge(accept_steps, "await_visualizations")
    workflow.add_edge("await_visualizations", END)

    return workflow.compile()
