"""Utility functions for the Financial Agent."""

import re
import sys
from typing import Tuple, Optional
from datetime import datetime
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.messages import BaseMessage

from frontier_challenge.tools.semantic_tool import SemanticSearchResult
from frontier_challenge.tools.sql_tool.models import FilterResult


class SafeParser(BaseOutputParser):
    """Safe parser to replace braces in messages for prompt template compatibility."""

    def parse(self, message):
        """Parse the message and replace braces."""
        if isinstance(message, BaseMessage):
            message.content = message.content.replace("{", "{{").replace("}", "}}")
        return message


def transform_roles(messages, as_string: bool = False, no_attachments: bool = False):
    """
    Transform message roles for LLM consumption.

    Parameters
    ----------
    messages : list
        List of messages to transform
    as_string : bool
        Whether to return as formatted string
    no_attachments : bool
        Whether to exclude attachments

    Returns
    -------
    list or str
        Transformed messages
    """
    if not messages:
        return "" if as_string else []

    transformed = []
    for msg in messages:
        role = msg.role if hasattr(msg, "role") else msg.type
        content = msg.content

        # Map roles
        if role in ["human", "user"]:
            role = "user"
        elif role in ["ai", "assistant", "fundai"]:
            role = "assistant"
        elif role == "system":
            role = "system"
        else:
            role = "assistant"

        if as_string:
            transformed.append(f"{role.capitalize()}: {content}")
        else:
            transformed.append((role, content))

    if as_string:
        return "\n".join(transformed)
    return transformed


def format_chat_history(messages: list) -> str:
    """
    Format message history for prompt context.

    Parameters
    ----------
    messages : list
        List of message objects

    Returns
    -------
    str
        Formatted chat history
    """
    if not messages:
        return "No previous conversation."

    formatted = []
    for msg in messages:
        role = msg.role if hasattr(msg, "role") else msg.type
        content = msg.content

        if role in ["user", "human"]:
            formatted.append(f"User: {content}")
        elif role in ["assistant", "ai", "fundai"]:
            formatted.append(f"FundAI: {content}")

    return "\n".join(formatted[-10:])  # Last 10 messages for context


def get_current_date() -> str:
    """Get current date formatted for prompts."""
    return datetime.now().strftime("%Y-%m-%d")


def detect_language(text: str) -> str:
    """
    Simple language detection (Portuguese vs English).

    Parameters
    ----------
    text : str
        Text to analyze

    Returns
    -------
    str
        "pt" for Portuguese, "en" for English
    """
    portuguese_indicators = [
        "olá",
        "oi",
        "fundos",
        "investimento",
        "retorno",
        "taxa",
        "brasileiro",
    ]

    text_lower = text.lower()
    pt_count = sum(1 for indicator in portuguese_indicators if indicator in text_lower)

    return "pt" if pt_count >= 2 else "en"


def format_tool_result(tool_name: str, result, language: str = "en") -> str:
    """
    Format tool results for user-friendly display.

    Parameters
    ----------
    tool_name : str
        Name of the tool that was executed
    result : SemanticSearchResult or FilterResult
        Result from the tool (Pydantic model)
    language : str
        User's preferred language

    Returns
    -------
    str
        Formatted result string
    """
    if tool_name == "semantic_search":
        return _format_semantic_result(result, language)
    elif tool_name == "structured_filter":
        return _format_filter_result(result, language)
    else:
        return str(result)


def _format_semantic_result(result: SemanticSearchResult, language: str) -> str:
    """Format semantic search results."""
    if not result.matches:
        if language == "pt":
            return "Nenhum fundo encontrado com os critérios especificados."
        return "No funds found matching your criteria."

    header = (
        f"Encontrei {len(result.matches)} fundo(s):\n\n"
        if language == "pt"
        else f"Found {len(result.matches)} fund(s):\n\n"
    )

    fund_list = []
    for i, match in enumerate(result.matches, 1):
        fund_info = f"{i}. **{match.legal_name}**\n"
        fund_info += f"   - CNPJ: {match.cnpj}\n"
        fund_info += f"   - Type: {match.fund_type}\n"
        fund_info += f"   - Relevance: {match.score:.2f}\n"
        fund_list.append(fund_info)

    return header + "\n".join(fund_list)


def _format_filter_result(result: FilterResult, language: str) -> str:
    """Format structured filter results."""
    # Handle error case
    if not result.success:
        error_prefix = "Erro" if language == "pt" else "Error"
        return f"{error_prefix}: {result.error_message or 'Unknown error'}"

    # Handle no results
    if not result.funds:
        if language == "pt":
            return "Nenhum fundo encontrado com os critérios especificados."
        return "No funds found matching your criteria."

    header = (
        f"Encontrei {len(result.funds)} fundo(s):\n\n"
        if language == "pt"
        else f"Found {len(result.funds)} fund(s):\n\n"
    )

    fund_list = []
    for i, fund in enumerate(result.funds[:10], 1):  # Limit to 10 for readability
        fund_info = f"{i}. **{fund.legal_name}**\n"
        fund_info += f"   - CNPJ: `{fund.cnpj}`\n"

        # Add classification if available
        if fund.investment_class:
            fund_info += f"   - Investment Class: {fund.investment_class}\n"
        if fund.anbima_classification:
            fund_info += f"   - ANBIMA Classification: {fund.anbima_classification}\n"
        if fund.fund_type:
            fund_info += f"   - Type: {fund.fund_type}\n"
        if fund.risk_class:
            fund_info += f"   - Risk Class: {fund.risk_class}\n"

        # Add ALL available performance metrics
        if fund.return_ytd_2024_avg is not None:
            fund_info += f"   - YTD 2024 Return: {fund.return_ytd_2024_avg:.2f}%\n"
        if fund.return_12m_avg is not None:
            fund_info += f"   - 12-Month Return: {fund.return_12m_avg:.2f}%\n"
        if fund.return_5y_pct is not None:
            fund_info += f"   - 5-Year Return: {fund.return_5y_pct:.2f}%\n"
        if fund.volatility_12m is not None:
            fund_info += f"   - Volatility (12M): {fund.volatility_12m:.2f}%\n"
        if fund.sharpe_ratio_approx is not None:
            fund_info += f"   - Sharpe Ratio: {fund.sharpe_ratio_approx:.2f}\n"

        # Add fee information
        if fund.management_fee_pct is not None:
            fund_info += f"   - Management Fee: {fund.management_fee_pct:.2f}%\n"
        if fund.performance_fee_pct is not None:
            fund_info += f"   - Performance Fee: {fund.performance_fee_pct:.2f}%\n"

        # Add size and constraints
        if fund.nav is not None:
            fund_info += f"   - NAV: R$ {fund.nav:,.2f}\n"
        if fund.min_initial_investment is not None:
            fund_info += f"   - Minimum Investment: R$ {fund.min_initial_investment:,.2f}\n"
        if fund.lockup_days is not None:
            fund_info += f"   - Lockup Period: {fund.lockup_days} days\n"

        fund_list.append(fund_info)

    if len(result.funds) > 10:
        more_msg = (
            f"\n... e mais {len(result.funds) - 10} fundos"
            if language == "pt"
            else f"\n... and {len(result.funds) - 10} more funds"
        )
        fund_list.append(more_msg)

    return header + "\n".join(fund_list)
