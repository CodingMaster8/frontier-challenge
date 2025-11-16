"""Main graph definition for the Financial Agent."""

import datetime
import logging
from io import BytesIO
from typing import Optional

import pandas as pd
from langchain_core.messages import AIMessage, ChatMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import START, END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import AnyMessage



from .models import AgentState, ToolReasoningResponse, VisualizationDecisionResponse
from .prompts import (
    FINANCIAL_AGENT_SYSTEM_PROMPT,
    TOOL_ROUTER_SYSTEM_PROMPT,
    GREETING_TEMPLATES,
    NEED_ANYTHING_ELSE_TEMPLATES,
    UNKNOWN_CAPABILITY_TEMPLATES,
    VISUALIZATION_DECISION_PROMPT,
)
from .utils import (
    transform_roles,
    get_current_date,
    detect_language,
    format_tool_result,
    _format_semantic_result,
    SafeParser,
)
from ..tools import SemanticSearchTool, StructuredFilterTool, FinancialVisualizationTool

logger = logging.getLogger(__name__)


def get_financial_agent_graph(
    db_path: str = "data/br_funds.db",
    model_name: str = "gpt-5.1",
    checkpointer: Optional[BaseCheckpointSaver] = None,
    session_state=None,
) -> CompiledStateGraph:
    """
    Build and compile the financial agent graph following expertai architecture.

    Parameters
    ----------
    db_path : str
        Path to DuckDB database
    model_name : str
        OpenAI model name
    checkpointer : Optional[BaseCheckpointSaver]
        Checkpoint saver for persistence
    session_state : optional
        Streamlit session state object

    Returns
    -------
    StateGraph
        Compiled agent graph
    """
    # Initialize LLMs
    llm_main = ChatOpenAI(model=model_name, temperature=0)
    llm_router = ChatOpenAI(model="gpt-5-chat-latest", temperature=0)

    # Initialize tools
    semantic_tool = SemanticSearchTool(db_path=db_path)
    filter_tool = StructuredFilterTool(db_path=db_path, refine_query=False)
    viz_tool = FinancialVisualizationTool(
        library="matplotlib",
        image_format="png",
        output_dir="output_visualizations",
        language="portuguese",
    )

    # Tool names and descriptions
    tool_names = ["semantic_search", "structured_filter"]

    logger.info(f"Registered tools: {tool_names}")

    def get_tools_description() -> str:
        """Get formatted description of available tools."""
        return """
            semantic_search: Search for funds using natural language query with semantic similarity. This tool uses vector embeddings to find funds based on conceptual similarity rather than exact keyword matches. It's ideal for exploratory searches and fuzzy matching scenarios. Use for natural language queries (e.g., "sustainable tech funds", "conservative bonds"), fuzzy/partial name matching (e.g., "Bradesco gold", "BB renda fixa"), conceptual searches (e.g., "low risk", "ESG investing"), queries in Portuguese or English, finding funds similar to a description or investment strategy, when exact CNPJ or legal name is unknown. Parameters: query (str, required), top_k (int, optional, default=10).

            structured_filter: Filter funds using natural language query that gets converted to SQL, or structured criteria. Use this for queries with specific numbers, comparisons, or precise filters like performance metrics (returns, volatility, Sharpe ratio), fees (management, performance, expense ratio), fund size/AUM, investment constraints, classification filters, risk class, or boolean filters. This tool converts natural language to SQL queries against the fund database. Parameters: query (str, optional), criteria (FundFilterCriteria, optional).
        """.strip()

    async def node_greeting(state: AgentState) -> dict:
        """Greet the user when starting a conversation."""
        language = state.user_language or "en"

        greeting = GREETING_TEMPLATES.get(language, GREETING_TEMPLATES["en"])

        logger.info("Sending greeting to user")

        return {
            "messages": [ChatMessage(content=greeting, role="fundai")],
            "internal_monologue": [ChatMessage(content=greeting, role="fundai")],
            "current_status": "ready",
        }

    async def node_await_user_input(state: AgentState) -> dict:
        """
        Receive user input and detect language.

        Captures user input through the message adapter and detects the language.
        """

        language = state.user_language or "en"

        #user_input = user_msg.content
        user_input = input("User: ")

        # Handle empty input
        if not user_input:
            logger.warning("Empty input received")
            return {
                "messages": [HumanMessage(content="")],
                "internal_monologue": [HumanMessage(content="")],
            }

        # Detect language only on first message
        if not state.user_language:
            detected_lang = detect_language(user_input)
            logger.info(f"Detected language: {detected_lang}")

            return {
                "user_language": detected_lang,
                "messages": [HumanMessage(content=user_input)],
                "internal_monologue": [HumanMessage(content=user_input)],
            }

        print(f"User input captured ({language}): {user_input}")
        return {
            "current_status": "analyzing_query",
            "messages": [HumanMessage(content=user_input)],
            "internal_monologue": [HumanMessage(content=user_input)],
        }

    async def node_reason_tool_call(state: AgentState) -> dict:
        """Reason about the user input and determine if a tool is needed."""
        user_message = state.messages[-1].content

        # Check if there was a previous error
        tool_invocation_has_error = (
            state.tool_invocation_has_error[-1]
            if state.tool_invocation_has_error
            else False
        )

        # Get error guidance if there was an error
        tool_invocation_error_guidance = ""
        if tool_invocation_has_error and state.tool_invocation_error_guidance:
            tool_invocation_error_guidance = state.tool_invocation_error_guidance[-1].content

        # Format chat history
        chat_history = transform_roles(
            state.internal_monologue[:-1] if state.internal_monologue else [],
            as_string=True,
            no_attachments=True,
        )

        # Build the prompt using ChatPromptTemplate
        prompt_template = ChatPromptTemplate.from_template(TOOL_ROUTER_SYSTEM_PROMPT)

        parser = PydanticOutputParser(pydantic_object=ToolReasoningResponse)

        # Build the chain
        chain = prompt_template | llm_router | parser

        # Prepare input for the chain
        values = {
            "tools_description": get_tools_description(),
            "chat_history": chat_history,
            "user_query": user_message,
            "format_instructions": parser.get_format_instructions(),
        }

        # If there was an error, we need to add it as additional messages
        if tool_invocation_has_error:
            # For error cases, we'll invoke the chain differently to include previous reasoning
            messages = [
                SystemMessage(content=TOOL_ROUTER_SYSTEM_PROMPT.format(**values))
            ]
            last_reasoning = state.user_message_tool_reasonings[-1]
            messages.append(last_reasoning)
            messages.append(HumanMessage(content=tool_invocation_error_guidance))

            result = await llm_router.ainvoke(messages)
            try:
                response = parser.parse(result.content)
            except Exception as e:
                logger.error(f"Failed to parse error retry response: {e}")
                # Fall back to storing the raw content
                response = ToolReasoningResponse(
                    tool_name="error",
                    tool_instruction="",
                    reasoning=result.content
                )
        else:
            # Normal case - use the chain (returns ToolReasoningResponse)
            response = await chain.ainvoke(values)

        logger.info(f"Tool reasoning: {response.reasoning[:100] if len(response.reasoning) > 100 else response.reasoning}")
        logger.info(f"Selected tool: {response.tool_name}")

        # Store the entire response as JSON for later extraction
        return {
            "user_message_tool_reasonings": [ChatMessage(content=response.model_dump_json(), role="fundai")],
        }

    async def node_extract_tool_call(state: AgentState) -> dict:
        """Extract tool name and instruction from reasoning."""
        reasoning_json = state.user_message_tool_reasonings[-1].content

        # Parse the ToolReasoningResponse from JSON
        try:
            response = ToolReasoningResponse.model_validate_json(reasoning_json)
            tool_name = response.tool_name.lower().strip()
            tool_instruction = response.tool_instruction
            reasoning = response.reasoning
        except Exception as e:
            logger.error(f"❌ Failed to parse ToolReasoningResponse: {e}")
            tool_name = "error"
            tool_instruction = ""
            reasoning = reasoning_json

        logger.info(f"🔧 Extracted tool: {tool_name}")
        logger.debug(f"Tool instruction: {tool_instruction[:200] if tool_instruction else '(empty)'}")

        return_dict = {
            "tool_invocations": [ChatMessage(content=tool_name, role="fundai")],
            "tool_instructions": [ChatMessage(content=tool_instruction, role="fundai")],
            "will_invoke_tool": [False],
            "tool_invocation_has_error": [False],
            "tool_invocation_error_guidance": [ChatMessage(content="", role="fundai")],
        }

        # Validate tool extraction
        if tool_name == "error":
            logger.error("❌ Tool extraction error: Invalid format")
            return_dict["tool_invocation_error_guidance"] = [
                ChatMessage(
                    content="Error: You didn't answer with the tool name and tool instructions in the correct format. Fix your output.",
                    role="fundai",
                )
            ]
            return_dict["tool_invocation_has_error"] = [True]

        elif tool_name not in tool_names + ["unknown_capability", "no_tool"]:
            logger.error(f"❌ Tool extraction error: Unknown tool '{tool_name}'")
            return_dict["tool_invocation_error_guidance"] = [
                ChatMessage(
                    content=f"Error: Tool '{tool_name}' does not exist. Pick a tool that exists, or reconsider your answer.",
                    role="fundai",
                )
            ]
            return_dict["tool_invocation_has_error"] = [True]

        elif tool_name not in ["unknown_capability", "no_tool"]:
            # Valid tool to invoke - set appropriate status
            logger.info(f"✅ Will invoke tool: {tool_name}")

            # Map tool names to status
            status_map = {
                "semantic_search": "searching_funds",
                "structured_filter": "filtering_data",
            }
            tool_status = status_map.get(tool_name, "processing_results")

            return_dict["will_invoke_tool"] = [True]
            return_dict["current_status"] = tool_status
            return_dict["internal_monologue"] = [
                ChatMessage(content=reasoning, role="fundai"),
                HumanMessage(content="Ok, proceed"),
            ]

        elif tool_name == "no_tool":
            # No tool needed, answer directly
            logger.info("No tool needed, answering directly")
            return_dict["current_status"] = "generating_response"
            return_dict["visualization_results"] = [[]]  # Clear previous visualizations
            return_dict["internal_monologue"] = [
                ChatMessage(content=reasoning, role="fundai"),
                HumanMessage(content="<think>"),
                ChatMessage(
                    content="Thought: I don't need any tools to answer this question. Time to answer!",
                    role="fundai",
                ),
                HumanMessage(content="Answer:"),
            ]

        elif tool_name == "unknown_capability":
            # Unknown capability
            logger.info("❓ Unknown capability detected")
            return_dict["current_status"] = "generating_response"
            return_dict["visualization_results"] = [[]]  # Clear previous visualizations
            return_dict["internal_monologue"] = [
                ChatMessage(content=reasoning, role="fundai"),
                HumanMessage(content="<think>"),
                ChatMessage(
                    content="Thought: I don't have this capability. I should inform the user.",
                    role="fundai",
                ),
                HumanMessage(content="Answer:"),
            ]

        return return_dict

    async def node_execute_semantic_search(state: AgentState) -> dict:
        """Execute semantic search tool."""
        instruction = state.tool_instructions[-1].content

        logger.info(f"Executing semantic search...")
        logger.debug(f"Query: {instruction}")

        try:
            # Build the index
            try:
                build_result = semantic_tool.build_index()
            except Exception as e:
                logger.error(f"❌ Error building semantic index: {e}", exc_info=True)

            # The output is a SemanticSearchResult object
            result = semantic_tool.semantic_search(query=instruction, top_k=5)

            logger.info(f"✅ Semantic search completed - found {len(result.matches) if hasattr(result, 'matches') else 0} results")

            formatted_result = _format_semantic_result(
                result, state.user_language
            )

            logger.debug(f"Formatted result preview: {formatted_result[:200]}...")

            return {
                "internal_monologue": [
                    ChatMessage(
                        content=f"Tool result:\n{formatted_result}",
                        role="tool",
                    )
                ],
                "should_answer_user": [True],
                "current_status": "processing_results",
                "visualization_results": [[]],  # Clear previous visualizations (semantic search doesn't generate viz)
            }

        except Exception as e:
            logger.error(f"❌ Error in semantic search: {e}", exc_info=True)
            error_msg = (
                f"Erro ao buscar fundos: {str(e)}"
                if state.user_language == "pt"
                else f"Error searching for funds: {str(e)}"
            )
            return {
                "internal_monologue": [
                    ChatMessage(content=f"Tool error: {error_msg}", role="tool")
                ],
                "should_answer_user": [True],
                "current_status": "error",
                "visualization_results": [[]],  # Clear previous visualizations on error
            }

    async def node_execute_structured_filter(state: AgentState) -> dict:
        """Execute structured filter tool."""
        instruction = state.tool_instructions[-1].content

        logger.info(f"📊 Executing structured filter...")
        logger.debug(f"Query: {instruction}")

        try:
            result = await filter_tool.structured_filter(query=instruction)

            logger.info(f"✅ Structured filter completed - found {len(result.funds)} records")
            logger.debug(f"Generated SQL: {result.sql_query}")

            # Pass the FilterResult object directly to format_tool_result
            formatted_result = format_tool_result(
                "structured_filter", result, state.user_language
            )

            logger.debug(f"Formatted result preview: {formatted_result[:200]}...")

            # Convert fund records to DataFrame for visualization
            df = None
            if result.funds and len(result.funds) > 0:
                # Convert list of FundRecord to DataFrame
                df = pd.DataFrame([fund.model_dump() for fund in result.funds])
                logger.info(f"Created DataFrame with shape {df.shape}")

            # Serialize DataFrame
            df_bytes = None
            if df is not None:
                buffer = BytesIO()
                df.to_pickle(buffer)
                df_bytes = buffer.getvalue()

            return {
                "internal_monologue": [
                    ChatMessage(
                        content=f"Tool result:\n{formatted_result}",
                        role="tool",
                    )
                ],
                "tool_result_dataframe": [df_bytes] if df_bytes else [],
                "should_answer_user": [True],
                "current_status": "processing_results",
            }

        except Exception as e:
            logger.error(f"❌ Error in structured filter: {e}", exc_info=True)
            error_msg = (
                f"Erro ao filtrar fundos: {str(e)}"
                if state.user_language == "pt"
                else f"Error filtering funds: {str(e)}"
            )
            return {
                "internal_monologue": [
                    ChatMessage(content=f"Tool error: {error_msg}", role="tool")
                ],
                "should_answer_user": [True],
                "current_status": "error",
            }

    async def node_answer_user_query(state: AgentState) -> dict:
        """Generate response to user based on context and tool results."""
        language = state.user_language
        logger.info(f"Generating answer to user query in {language}...")

        # Build system prompt
        system_prompt = FINANCIAL_AGENT_SYSTEM_PROMPT.format(
            language="Portuguese" if language == "pt" else "English",
            date=get_current_date(),
        )

        # Build conversation context
        messages = [
            SystemMessage(content=system_prompt),
        ]

        # Add conversation history - limit to last 10 messages for efficiency
        internal_monologue = state.internal_monologue or []
        recent_monologue = internal_monologue[-10:] if len(internal_monologue) > 10 else internal_monologue

        if len(internal_monologue) > 10:
            logger.debug(f"Limiting context window: using last 10 of {len(internal_monologue)} messages")

        messages.extend(transform_roles(recent_monologue))

        # Build prompt
        prompt = ChatPromptTemplate.from_messages(messages)

        # Generate response
        response = await llm_main.ainvoke(prompt.format_messages())
        response_content = response.content

        logger.info(f"Generated response: {response_content[:100]}...")

        # Don't send through adapter - Streamlit app handles this

        return {
            "messages": [ChatMessage(content=response_content, role="fundai")],
            "internal_monologue": [ChatMessage(content=response_content, role="fundai")],
            "should_answer_user": [False],
            "current_status": "ready",
        }

    async def node_unknown_capability(state: AgentState) -> dict:
        """Handle queries outside agent capabilities."""
        language = state.user_language

        message = UNKNOWN_CAPABILITY_TEMPLATES.get(
            language, UNKNOWN_CAPABILITY_TEMPLATES["en"]
        )

        logger.info("Handling unknown capability")

        return {
            "messages": [ChatMessage(content=message, role="fundai")],
            "internal_monologue": [ChatMessage(content=message, role="fundai")],
        }

    async def node_decide_visualization(state: AgentState) -> dict:
        """Decide if visualization should be generated based on results."""
        user_message = state.messages[-1].content if len(state.messages) > 0 else ""
        language = state.user_language
        tool_name = state.tool_invocations[-1].content

        logger.info("🎨 Deciding if visualization is needed...")

        # Only consider visualization for structured_filter tool
        if tool_name != "structured_filter":
            logger.info("❌ Tool is not structured_filter, skipping visualization")
            return {
                "should_generate_visualization": [False],
                "visualization_reasoning": ["Visualization only supported for structured_filter results"],
                "visualization_results": [[]],  # Clear previous visualizations
            }

        # Check if we have tool result dataframe
        if not state.tool_result_dataframe or len(state.tool_result_dataframe) == 0:
            logger.info("❌ No dataframe available for visualization")
            return {
                "should_generate_visualization": [False],
                "visualization_reasoning": ["No data available for visualization"],
                "visualization_results": [[]],  # Clear previous visualizations
            }

        # Get result count from dataframe
        df_bytes = state.tool_result_dataframe[-1]
        df = pd.read_pickle(BytesIO(df_bytes))
        result_count = len(df)

        logger.info(f"Result count: {result_count}")

        # Build decision prompt with structured output
        prompt_template = ChatPromptTemplate.from_template(VISUALIZATION_DECISION_PROMPT)

        # Determine query type
        query_type = "filter_query" if "filter" in tool_name.lower() else "search_query"
        data_type = "tabular_fund_data"

        # Create parser for structured output
        parser = PydanticOutputParser(pydantic_object=VisualizationDecisionResponse)

        # Build the chain with parser
        chain = prompt_template | llm_router | parser

        # Invoke decision with format instructions
        try:
            response = await chain.ainvoke({
                "user_query": user_message,
                "result_count": result_count,
                "data_type": data_type,
                "query_type": query_type,
                "format_instructions": parser.get_format_instructions(),
            })

            should_visualize = response.should_visualize
            reasoning = response.reasoning

        except Exception as e:
            logger.error(f"❌ Failed to parse visualization decision: {e}")
            logger.debug(f"Raw response: {e}")
            should_visualize = False
            reasoning = f"Error parsing decision: {str(e)}"

        logger.info(f"{'✅' if should_visualize else '❌'} Visualization decision: {should_visualize}")
        logger.info(f"💭 Reasoning: {reasoning}")

        status = "generating_visualization" if should_visualize else "processing_results"

        result = {
            "should_generate_visualization": [should_visualize],
            "visualization_reasoning": [reasoning],
            "current_status": status,
        }

        # Clear previous visualizations if not generating new ones
        if not should_visualize:
            result["visualization_results"] = [[]]

        return result

    async def node_generate_visualization(state: AgentState) -> dict:
        """Generate visualization from structured filter results."""
        logger.info("🎨 Generating visualization...")

        try:
            # Get the dataframe from state
            df_bytes = state.tool_result_dataframe[-1]
            df = pd.read_pickle(BytesIO(df_bytes))

            # Get user query for context
            user_message = state.messages[-1].content if len(state.messages) > 0 else ""

            # Get language
            language = state.user_language or "pt"
            viz_language = "portuguese" if language == "pt" else "english"

            # Update viz tool language if needed
            viz_tool.language = viz_language

            logger.info(f"📊 Generating visualization for {len(df)} results")
            logger.debug(f"DataFrame columns: {df.columns.tolist()}")

            # Generate visualization
            viz_results = await viz_tool.create_visualization(
                data=df,
                query=user_message,
                number_visualizations=1,
                library="matplotlib",
                image_format="png",
            )

            logger.info(f"✅ Generated {len(viz_results)} visualization(s)")

            # Convert to serializable format
            viz_list = []
            for viz in viz_results:
                viz_list.append({
                    "image_path": viz.image_path,
                    "description": viz.description,
                    "visualization_type": viz.visualization_type,
                    "python_code": viz.python_code,
                })

            # Add visualization info to internal monologue
            viz_summary = f"\n\n[Visualization generated: {viz_results[0].visualization_type}]"

            return {
                "visualization_results": [viz_list],
                "internal_monologue": [ChatMessage(content=viz_summary, role="system")],
                "current_status": "generating_visualization",
            }

        except Exception as e:
            logger.error(f"❌ Error generating visualization: {e}", exc_info=True)
            error_msg = f"Visualization generation failed: {str(e)}"
            return {
                "visualization_results": [[]],
                "internal_monologue": [ChatMessage(content=f"[{error_msg}]", role="system")],
                "current_status": "generating_visualization",
            }

    # ============================================================================
    # Workflow Routing Functions
    # ============================================================================

    def tool_call_router(state: AgentState) -> str:
        """Route to the appropriate node based on tool decision."""
        will_invoke_tool = state.will_invoke_tool[-1]
        tool_name = state.tool_invocations[-1].content
        has_error = state.tool_invocation_has_error[-1] if state.tool_invocation_has_error else False

        if will_invoke_tool:
            return tool_name
        elif tool_name == "no_tool":
            return "answer_user_query"
        elif tool_name == "unknown_capability":
            return "unknown_capability"

        # Error case - prevent infinite loops with retry counter
        if has_error:
            # Count consecutive errors in the last 3 attempts
            error_count = sum(1 for e in state.tool_invocation_has_error[-3:] if e)
            if error_count >= 2:
                logger.warning("Too many tool extraction errors, answering directly")
                return "answer_user_query"

        # Retry reasoning
        return "reason_tool_call"

    def post_tool_router(state: AgentState) -> str:
        """Route after tool execution."""
        should_answer_user = state.should_answer_user[-1]
        tool_name = state.tool_invocations[-1].content

        # For structured_filter, check if visualization is needed
        if tool_name == "structured_filter" and should_answer_user:
            return "decide_visualization"

        if should_answer_user:
            return "answer_user_query"

        return "end_turn"

    def visualization_router(state: AgentState) -> str:
        """Route after deciding visualization."""
        should_generate_visualization = state.should_generate_visualization[-1]

        if should_generate_visualization:
            return "generate_visualization"
        else:
            return "answer_user_query"

    def post_visualization_router(state: AgentState) -> str:
        """Route after visualization generation."""
        # Always answer user after visualization
        return "answer_user_query"

    # ============================================================================
    # Workflow Builder
    # ============================================================================

    # Build the graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("greeting", node_greeting)
    workflow.add_node("await_user_input", node_await_user_input)
    workflow.add_node("reason_tool_call", node_reason_tool_call)
    workflow.add_node("extract_tool_call", node_extract_tool_call)
    workflow.add_node("semantic_search", node_execute_semantic_search)
    workflow.add_node("structured_filter", node_execute_structured_filter)
    workflow.add_node("decide_visualization", node_decide_visualization)
    workflow.add_node("generate_visualization", node_generate_visualization)
    workflow.add_node("answer_user_query", node_answer_user_query)
    workflow.add_node("unknown_capability", node_unknown_capability)

    def entry_router(state: AgentState) -> str:
        """Route entry based on whether we have a user message."""
        # If we have messages, skip greeting and go straight to reasoning
        if state.messages and len(state.messages) > 0:
            return "reason_tool_call"
        # Otherwise, start with greeting
        return "greeting"

    # Add edges
    workflow.add_conditional_edges(START, entry_router)
    workflow.add_edge("greeting", "await_user_input")

    # Main conversation flow
    workflow.add_edge("await_user_input", "reason_tool_call")
    workflow.add_edge("reason_tool_call", "extract_tool_call")

    # After extracting tool call, route based on decision
    workflow.add_conditional_edges("extract_tool_call", tool_call_router)

    # Tool nodes route to post_tool_router
    workflow.add_conditional_edges("semantic_search", post_tool_router)
    workflow.add_conditional_edges("structured_filter", post_tool_router)

    # Visualization flow
    workflow.add_conditional_edges("decide_visualization", visualization_router)
    workflow.add_conditional_edges("generate_visualization", post_visualization_router)

    # Other routes
    workflow.add_edge("unknown_capability", "answer_user_query")
    workflow.add_edge("answer_user_query", END)  # End after answering for Streamlit


    # Compile graph
    compiled_workflow = workflow.compile(checkpointer=checkpointer)

    logger.info("Financial agent graph compiled successfully")

    return compiled_workflow
