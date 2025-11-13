"""
Simple tests for SQL tool using programmatic criteria (no LLM needed)
"""
import pytest
from frontier_challenge.tools.sql_tool import StructuredFilterTool
from frontier_challenge.tools.sql_tool.models import FundFilterCriteria


@pytest.fixture
def sql_tool():
    """Create a StructuredFilterTool instance"""
    return StructuredFilterTool(db_path="data/br_funds.db")


def test_filter_with_min_return_ytd(sql_tool):
    """Test filtering funds by minimum YTD return"""
    criteria = FundFilterCriteria(
        min_return_ytd=10.0,
        limit=5
    )

    result = sql_tool.filter_funds(criteria=criteria)

    assert result.success is True
    assert result.total_count <= 5
    assert result.sql_query is not None
    assert "return_ytd_2024_avg > 10.0" in result.sql_query

    # Check that all returned funds meet the criteria
    for fund in result.funds:
        if fund.return_ytd_2024_avg is not None:
            assert fund.return_ytd_2024_avg > 10.0


def test_filter_with_max_management_fee(sql_tool):
    """Test filtering funds by maximum management fee"""
    criteria = FundFilterCriteria(
        max_management_fee=2.0,
        limit=10
    )

    result = sql_tool.filter_funds(criteria=criteria)

    assert result.success is True
    assert result.sql_query is not None
    assert "management_fee_pct < 2.0" in result.sql_query

    # Check that all returned funds meet the criteria
    for fund in result.funds:
        if fund.management_fee_pct is not None:
            assert fund.management_fee_pct < 2.0


def test_filter_with_min_nav(sql_tool):
    """Test filtering funds by minimum net asset value"""
    criteria = FundFilterCriteria(
        min_nav=100_000_000,  # R$ 100M
        limit=5
    )

    result = sql_tool.filter_funds(criteria=criteria)

    assert result.success is True
    assert result.sql_query is not None
    assert "nav > 100000000" in result.sql_query


def test_filter_with_multiple_criteria(sql_tool):
    """Test filtering with multiple criteria combined"""
    criteria = FundFilterCriteria(
        min_return_ytd=5.0,
        max_management_fee=2.5,
        min_nav=50_000_000,
        limit=10
    )

    result = sql_tool.filter_funds(criteria=criteria)

    assert result.success is True
    assert result.sql_query is not None
    assert "return_ytd_2024_avg > 5.0" in result.sql_query
    assert "management_fee_pct < 2.5" in result.sql_query
    assert "nav > 50000000" in result.sql_query


def test_filter_with_no_criteria_returns_error(sql_tool):
    """Test that filtering without query or criteria returns error"""
    result = sql_tool.filter_funds()

    assert result.success is False
    assert result.error_message is not None
    assert "Must provide query, criteria, or template" in result.error_message


def test_result_has_execution_time(sql_tool):
    """Test that results include execution time"""
    criteria = FundFilterCriteria(limit=5)

    result = sql_tool.filter_funds(criteria=criteria)

    assert result.success is True
    assert result.execution_time_ms is not None
    assert result.execution_time_ms > 0
