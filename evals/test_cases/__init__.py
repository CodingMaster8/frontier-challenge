"""Test case registry and loader."""

from typing import Dict, List
from ..models import ToolType
from ..evaluator import EvalSuite

from .semantic_search_cases import get_semantic_search_suite
from .structured_filter_cases import get_structured_filter_suite
from .holdings_search_cases import get_holdings_search_suite
from .agent_routing_cases import get_agent_routing_suite


def get_all_suites() -> Dict[ToolType, EvalSuite]:
    """Get all evaluation suites organized by tool type."""
    return {
        ToolType.SEMANTIC_SEARCH: get_semantic_search_suite(),
        ToolType.STRUCTURED_FILTER: get_structured_filter_suite(),
        ToolType.HOLDINGS_SEARCH: get_holdings_search_suite(),
        ToolType.AGENT_ROUTING: get_agent_routing_suite(),
    }


def get_suite_by_tool(tool_type: ToolType) -> EvalSuite:
    """Get evaluation suite for a specific tool."""
    suites = get_all_suites()
    return suites.get(tool_type)


def get_all_cases() -> List:
    """Get all test cases across all suites."""
    suites = get_all_suites()
    all_cases = []
    for suite in suites.values():
        all_cases.extend(suite.cases)
    return all_cases


def get_edge_cases() -> List:
    """Get all edge cases across all suites."""
    all_cases = get_all_cases()
    return [case for case in all_cases if case.edge_case]


def get_cases_by_tags(tags: List[str]) -> List:
    """Get cases matching any of the provided tags."""
    all_cases = get_all_cases()
    return [
        case for case in all_cases
        if any(tag in case.tags for tag in tags)
    ]


__all__ = [
    "get_all_suites",
    "get_suite_by_tool",
    "get_all_cases",
    "get_edge_cases",
    "get_cases_by_tags",
]
