"""Pydantic models for the Financial Agent."""

import operator
from typing import Annotated, Optional, Sequence

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field


class AgentState(BaseModel):
    """Base state for the financial agent conversation flow."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Core conversation state
    messages: Annotated[Sequence[AnyMessage], add_messages] = Field(
        default=None,
        description="Messages visible to the user in the conversation. These are the primary chat messages exchanged between the user and the agent."
    )

    internal_monologue: Annotated[Sequence[AnyMessage], add_messages] = Field(
        default=None,
        description="Internal reasoning messages not shown to the user. Used for agent's thought process, decision-making steps, and debugging information."
    )

    # Tool routing state
    tool_instructions: Annotated[Sequence[AnyMessage], add_messages] = Field(
        default=None,
        description="Natural language instructions to pass to tools. Each instruction should clearly describe what the tool needs to accomplish in plain language."
    )

    tool_invocations: Annotated[Sequence[AnyMessage], add_messages] = Field(
        default=None,
        description="Names of tools that have been invoked during the conversation. Used to track which tools were called and in what order."
    )

    tool_invocation_error_guidance: Annotated[Sequence[AnyMessage], add_messages] = Field(
        default=None,
        description="Error messages and guidance when tool invocations fail. Helps the agent understand what went wrong and how to recover."
    )

    user_message_tool_reasonings: Annotated[Sequence[AnyMessage], add_messages] = Field(
        default=None,
        description="Agent's reasoning for selecting specific tools based on user messages. Explains why certain tools were chosen over others."
    )

    tool_invocation_has_error: Annotated[Sequence[bool], operator.add] = Field(
        default=None,
        description="Boolean flags indicating whether each tool invocation resulted in an error. True means error occurred, False means success."
    )

    will_invoke_tool: Annotated[Sequence[bool], operator.add] = Field(
        default=None,
        description="Boolean flags indicating whether a tool will be invoked in the current step. True means tool call is planned, False means no tool needed."
    )

    should_answer_user: Annotated[Sequence[bool], operator.add] = Field(
        default=None,
        description="Boolean flags indicating whether the agent should respond to the user after tool execution. True means provide answer, False means continue processing."
    )

    # Session management
    session_summary: Annotated[Sequence[str], operator.add] = Field(
        default=None,
        description="Brief summaries of conversation sessions. Used to maintain context across multiple interactions and remember key discussion points."
    )

    user_language: str = Field(
        default="en",
        description="User's preferred language code. Supports 'en' (English) and 'pt' (Portuguese). All responses should be in this language."
    )

    current_status: str = Field(
        default="idle",
        description="Current status of the agent execution. Used to show informative status messages in the UI. Examples: 'greeting', 'analyzing_query', 'searching_funds', 'filtering_data', 'generating_response'."
    )

    langfuse_post_interrupt_tags: list[str] = Field(
        default=None,
        description="Tags to add to Langfuse trace after human-in-the-loop interrupts. Used for monitoring and debugging agent behavior."
    )


class ToolReasoningResponse(BaseModel):
    """Response from tool reasoning node."""

    tool_name: str = Field(
        description="Name of the tool to invoke. Use 'no_tool' if no tool is needed, 'unknown_capability' if the request is outside agent capabilities. Examples: 'semantic_search', 'structured_filter', 'portfolio_analysis'."
    )

    tool_instruction: str = Field(
        description="Clear, natural language instruction for the tool explaining what it should do. Be specific about the query, filters, or analysis needed. Example: 'Search for Brazilian equity funds with high returns' or 'Filter funds by Sharpe ratio > 1.5'."
    )

    reasoning: str = Field(
        description="Internal reasoning explaining why this tool was selected and how it will help answer the user's query. Include: (1) What the user is asking for, (2) Why this tool is the best choice, (3) What information the tool will provide."
    )


class SemanticSearchState(AgentState):
    """State for semantic search tool execution."""

    search_query: str = Field(
        default=None,
        description="Natural language search query for semantic search. Should describe the type of funds or financial products the user is looking for. Example: 'equity funds with focus on technology sector' or 'conservative fixed income funds'."
    )

    search_results: list[dict] = Field(
        default=None,
        description="List of fund results returned from semantic search. Each dict contains fund details like CNPJ, fund name, type, administrator, and similarity score. Ordered by relevance to the query."
    )

    search_execution_time: float = Field(
        default=None,
        description="Time taken to execute the semantic search in seconds. Used for performance monitoring and optimization."
    )


class StructuredFilterState(AgentState):
    """State for structured filter tool execution."""

    filter_query: str = Field(
        default=None,
        description="Natural language filter query describing the filtering criteria. Should specify metrics, thresholds, and conditions. Example: 'funds with Sharpe ratio above 2 and returns greater than 10%' or 'equity funds with AUM over 100 million'."
    )

    generated_sql: str = Field(
        default=None,
        description="SQL query generated from the natural language filter query. This is the executable SQL that will be run against the database to retrieve matching funds based on structured criteria."
    )

    filter_results: list[dict] = Field(
        default=None,
        description="List of fund results matching the filter criteria. Each dict contains fund details and the specific metrics that were filtered on. Results satisfy all conditions specified in the filter query."
    )

    filter_execution_time: float = Field(
        default=None,
        description="Time taken to execute the structured filter query in seconds. Includes SQL generation and execution time. Used for performance monitoring."
    )
