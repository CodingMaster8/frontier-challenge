"""
LangGraph workflow for text-to-SQL conversion.

This module implements a workflow for converting natural language
queries into SQL queries, with multiple validation steps and automatic error recovery.
"""

import logging

from langchain_core.prompts.chat import ChatPromptTemplate
from langgraph.graph import END, START, StateGraph
from langchain_openai import ChatOpenAI

from .models import FilterQueryState
from .structured_filter_prompts import (
    SQL_GENERATION_PROMPT,
    SQL_REFINEMENT_PROMPT,
    ERROR_FIX_PROMPT,
)
from .utils import validate_sql_syntax, execute_sql_safe, extract_sql_from_markdown
from frontier_challenge.settings import OPENAI_API_KEY

logger = logging.getLogger(__name__)


# ============================================================================
# Workflow Nodes
# ============================================================================

def get_graph(db: str, max_retries: int) -> StateGraph:
    """Build and return the filter workflow graph"""

    llm_query_gen = ChatOpenAI(
        model="o4-mini-2025-04-16", openai_api_key=OPENAI_API_KEY
    ).with_retry()

    llm_light = ChatOpenAI(
        model="gpt-5-chat-latest", openai_api_key=OPENAI_API_KEY
    ).with_retry()

    workflow = StateGraph(FilterQueryState)


    async def node_generate_sql(state: FilterQueryState) -> dict:
        """Generate initial SQL query from natural language"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", SQL_GENERATION_PROMPT),
            ("user", f"""
            Schema:
            {state.view_schema}

            Natural language query: {state.natural_language_query}

            Generate a SQL query to answer this question.
            """)
        ])

        chain = prompt | llm_query_gen
        response = chain.invoke({})

        # Extract SQL from response
        sql = extract_sql_from_markdown(response.content)

        logger.info(f"Generated SQL: {sql[:200]}...")

        return {
            "generated_sql": sql,
            "messages": [response]
        }


    async def node_validate_syntax(state: FilterQueryState) -> dict:
        """Validate SQL syntax"""

        sql = state.generated_sql if state.generated_sql else state.refined_sql

        is_valid, error_msg = validate_sql_syntax(sql, db)

        if is_valid:
            logger.info("SQL syntax is valid")
            return {
                "error_message": None
            }
        else:
            logger.warning(f"SQL syntax error: {error_msg}")
            return {
                "error_message": error_msg,
                "retry_count": state.retry_count + 1
            }


    async def node_refine_query(state: FilterQueryState) -> dict:
        """Refine the SQL query for correctness and efficiency"""

        sql = state.generated_sql

        prompt = ChatPromptTemplate.from_messages([
            ("system", SQL_REFINEMENT_PROMPT),
            ("user", f"""
            Schema:
            {state.view_schema}

            Original query request: {state.natural_language_query}

            Generated SQL:
            {sql}

            Review this query and provide an improved version if needed. Make sure it:
            1. Uses correct column names from the schema
            2. Has proper WHERE clauses
            3. Includes appropriate ORDER BY
            4. Has reasonable LIMIT
            5. Uses correct DuckDB syntax
            """)
        ])

        chain = prompt | llm_light
        response = chain.invoke({})

        # Extract refined SQL
        refined_sql = extract_sql_from_markdown(response.content)

        logger.info(f"Refined SQL: {refined_sql[:200]}...")

        return {
            "refined_sql": refined_sql,
            "messages": state.messages + [response]
        }


    async def node_fix_error(state: FilterQueryState) -> dict:
        """Fix SQL error based on error message"""

        sql = state.refined_sql if state.refined_sql else state.generated_sql
        error = state.error_message

        prompt = ChatPromptTemplate.from_messages([
            ("system", ERROR_FIX_PROMPT),
            ("user", f"""
            Schema:
            {state.view_schema}

            Original query request: {state.natural_language_query}

            Current SQL (with error):
            {sql}

            Error message:
            {error}

            Fix the SQL query to resolve this error.
            """)
        ])

        chain = prompt | llm_query_gen
        response = chain.invoke({})

        # Extract fixed SQL
        fixed_sql = extract_sql_from_markdown(response.content)

        logger.info(f"Fixed SQL: {fixed_sql[:200]}...")

        return {
            "refined_sql": fixed_sql,
            "messages": state.messages + [response]
        }


    async def node_execute_query(state: FilterQueryState) -> dict:
        """Execute the SQL query"""

        sql = state.refined_sql if state.refined_sql else state.generated_sql

        success, message = execute_sql_safe(sql, db)

        if success:
            logger.info(message)
            return {
                "final_sql": sql,
                "error_message": None
            }
        else:
            logger.error(message)
            return {
                "error_message": message,
                "retry_count": state.retry_count + 1
            }


    async def node_finalize(state: FilterQueryState) -> dict:
        """Finalize the workflow"""
        sql = state.refined_sql if state.refined_sql else state.generated_sql
        return {
            "final_sql": sql
        }


    async def node_too_many_retries(state: FilterQueryState) -> dict:
        """Handle maximum retry limit reached"""
        error_msg = f"Maximum retries ({max_retries}) exceeded. Could not generate valid SQL."
        logger.error(error_msg)
        return {
            "error_message": error_msg,
            "final_sql": None
        }

    # ============================================================================
    # Workflow Routing
    # ============================================================================


    def route_after_validation(state: FilterQueryState) -> str:
        """Route after syntax validation"""
        if state.error_message is None:
            return "refine_query"
        elif state.retry_count >= max_retries:
            return "too_many_retries"
        else:
            return "fix_error"


    def route_after_refined_validation(state: FilterQueryState) -> str:
        """Route after refined syntax validation"""
        if state.error_message is None:
            return "execute_query"
        elif state.retry_count >= max_retries:
            return "too_many_retries"
        else:
            return "fix_error"


    def route_after_execution(state: FilterQueryState) -> str:
        """Route after query execution"""
        if state.error_message is None:
            return "finalize"
        elif state.retry_count >= max_retries:
            return "too_many_retries"
        else:
            return "fix_error"


    # ============================================================================
    # Workflow Builder
    # ============================================================================

    # Add nodes
    workflow.add_node(
        "generate_sql",
        node_generate_sql
    )

    workflow.add_node(
        "validate_syntax",
        node_validate_syntax
    )

    workflow.add_node(
        "validate_syntax_refined",
        node_validate_syntax
    )

    workflow.add_node(
        "refine_query",
        node_refine_query
    )

    workflow.add_node(
        "fix_error",
        node_fix_error
    )

    workflow.add_node(
        "execute_query",
        node_execute_query
    )

    workflow.add_node(
        "finalize",
        node_finalize
    )

    workflow.add_node(
        "too_many_retries",
        node_too_many_retries
    )

    # Define edges
    workflow.add_edge(START, "generate_sql")
    workflow.add_edge("generate_sql", "validate_syntax")

    # Route after initial validation
    workflow.add_conditional_edges(
        "validate_syntax",
        route_after_validation,
    )

    # Route after refinement
    workflow.add_edge("refine_query", "validate_syntax_refined")

    # Route after refined validation
    workflow.add_conditional_edges(
        "validate_syntax_refined",
        route_after_refined_validation,
    )

    # Route after fixing error
    workflow.add_edge("fix_error", "validate_syntax_refined")

    # Route after execution
    workflow.add_conditional_edges(
        "execute_query",
        route_after_execution,
    )

    # Finalize and end
    workflow.add_edge("finalize", END)
    workflow.add_edge("too_many_retries", END)

    # Compile workflow
    return workflow.compile()
