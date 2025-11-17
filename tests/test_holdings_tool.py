"""
Simple test script for the Holdings Search Tool.

Run with: python -m tests.test_holdings_tool
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from frontier_challenge.tools import HoldingsSearchTool
from frontier_challenge.tools.holdings_tool.models import HoldingsSearchCriteria


async def test_basic_search():
    """Test basic company name search"""
    print("=" * 80)
    print("TEST 1: Basic Company Name Search (Petrobras)")
    print("=" * 80)

    tool = HoldingsSearchTool(db_path="data/br_funds.db")

    result = await tool.search_holdings(company_name="Petrobras")

    print(f"✓ Success: {result.success}")
    print(f"✓ Search Method: {result.search_method}")
    print(f"✓ Total Results: {result.total_count}")
    print(f"✓ Unique Funds: {result.unique_funds_count}")
    print(f"✓ Execution Time: {result.execution_time_ms:.2f}ms")

    if result.error_message:
        print(f"✗ Error: {result.error_message}")
        return False

    # Show first few results
    if result.holdings:
        print(f"\nFirst 3 holdings:")
        for i, holding in enumerate(result.holdings[:3], 1):
            print(f"  {i}. {holding.legal_name[:50]}")
            print(f"     Asset: {holding.asset_name}")
            print(f"     Weight: {holding.portfolio_weight_pct:.2f}%")

    return True


async def test_fuzzy_matching():
    """Test fuzzy matching with Levenshtein distance"""
    print("\n" + "=" * 80)
    print("TEST 2: Fuzzy Matching Search (Vale)")
    print("=" * 80)

    tool = HoldingsSearchTool(db_path="data/br_funds.db")

    criteria = HoldingsSearchCriteria(
        company_name="Vale",
        use_fuzzy_match=True,
        min_similarity=0.7,
        group_by_fund=True,
        limit=10
    )

    result = await tool.search_holdings(criteria=criteria)

    print(f"✓ Success: {result.success}")
    print(f"✓ Search Method: {result.search_method}")
    print(f"✓ Unique Funds: {result.unique_funds_count}")

    if result.fund_summaries:
        print(f"\nTop 5 funds holding Vale:")
        for i, summary in enumerate(result.fund_summaries[:5], 1):
            print(f"  {i}. {summary.legal_name[:50]}")
            print(f"     Asset: {summary.asset_name}")
            print(f"     Weight: {summary.portfolio_weight_pct:.2f}%")

    return True


async def test_natural_language():
    """Test natural language query"""
    print("\n" + "=" * 80)
    print("TEST 3: Natural Language Query")
    print("=" * 80)

    tool = HoldingsSearchTool(db_path="data/br_funds.db")

    result = await tool.search_holdings(query="Funds that invest in Itau")

    print(f"✓ Success: {result.success}")
    print(f"✓ Unique Funds: {result.unique_funds_count}")

    if result.fund_summaries:
        print(f"\nTop 3 funds:")
        for i, summary in enumerate(result.fund_summaries[:3], 1):
            print(f"  {i}. {summary.legal_name[:50]}")
            print(f"     Asset: {summary.asset_name}")

    return True


async def test_with_filters():
    """Test search with additional filters"""
    print("\n" + "=" * 80)
    print("TEST 4: Search with Filters (Equity + Min Weight)")
    print("=" * 80)

    tool = HoldingsSearchTool(db_path="data/br_funds.db")

    criteria = HoldingsSearchCriteria(
        company_name="Banco",
        use_fuzzy_match=True,
        min_similarity=0.6,
        asset_class="EQUITY",
        min_weight=2.0,
        group_by_fund=True,
        limit=10
    )

    result = await tool.search_holdings(criteria=criteria)

    print(f"✓ Success: {result.success}")
    print(f"✓ Unique Funds: {result.unique_funds_count}")
    print(f"✓ Execution Time: {result.execution_time_ms:.2f}ms")

    if result.fund_summaries:
        print(f"\nFunds with >2% weight in bank stocks:")
        for i, summary in enumerate(result.fund_summaries[:5], 1):
            print(f"  {i}. {summary.legal_name[:50]}")
            print(f"     Asset: {summary.asset_name}")
            print(f"     Weight: {summary.portfolio_weight_pct:.2f}%")

    return True


async def test_sql_generation():
    """Test SQL query generation"""
    print("\n" + "=" * 80)
    print("TEST 5: SQL Query Generation")
    print("=" * 80)

    tool = HoldingsSearchTool(db_path="data/br_funds.db")

    criteria = HoldingsSearchCriteria(
        company_name="Petrobras",
        use_fuzzy_match=True,
        min_similarity=0.7,
        asset_class="EQUITY",
        group_by_fund=True,
        limit=5
    )

    result = await tool.search_holdings(criteria=criteria)

    print("✓ Generated SQL Query:")
    print("-" * 80)
    print(result.sql_query)
    print("-" * 80)

    return True


async def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("HOLDINGS SEARCH TOOL - TEST SUITE")
    print("=" * 80 + "\n")

    tests = [
        test_basic_search,
        test_fuzzy_matching,
        test_natural_language,
        test_with_filters,
        test_sql_generation,
    ]

    results = []
    for test in tests:
        try:
            success = await test()
            results.append(success)
        except Exception as e:
            print(f"\n✗ Test failed with error: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Tests passed: {passed}/{total}")

    if passed == total:
        print("✓ All tests passed!")
    else:
        print(f"✗ {total - passed} test(s) failed")

    return passed == total


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
