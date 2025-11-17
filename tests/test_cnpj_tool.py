"""
Quick test script for CNPJ Lookup Tool
"""

import asyncio
from frontier_challenge.tools.cnpj_tool import CNPJLookupTool

def test_cnpj_tool():
    """Test the CNPJ lookup tool with sample data"""

    print("Testing CNPJ Lookup Tool\n")
    print("=" * 80)

    # Initialize tool
    tool = CNPJLookupTool(db_path="data/br_funds.db")

    # Test 1: Get a sample CNPJ from the database first
    print("\nTest 1: Finding a sample CNPJ from database...")
    import duckdb
    conn = duckdb.connect("data/br_funds.db")
    sample = conn.execute("""
        SELECT cnpj, legal_name
        FROM fund_semantic_search_view
        WHERE cnpj IS NOT NULL
        LIMIT 3
    """).fetchdf()
    conn.close()

    if sample.empty:
        print("No funds found in database")
        return

    print(f"Found {len(sample)} sample funds:")
    for idx, row in sample.iterrows():
        print(f"   {idx+1}. {row['cnpj']} - {row['legal_name']}")

    # Test 2: Single CNPJ lookup
    print("\nTest 2: Single CNPJ lookup...")
    test_cnpj = sample['cnpj'].iloc[0]
    print(f"Looking up: {test_cnpj}")

    result = tool.lookup_by_cnpj(test_cnpj)

    if result.success:
        print(f"Success! Found {result.total_count} fund(s) in {result.execution_time_ms:.2f}ms")
        for fund in result.funds:
            print(f"\n   {fund.legal_name}")
            print(f"      CNPJ: {fund.cnpj}")
            print(f"      Type: {fund.fund_type}")
            print(f"      Net Asset Value: R$ {fund.net_asset_value:,.2f}" if fund.net_asset_value else "      Net Asset Value: N/A")
            print(f"      Management Fee: {fund.management_fee_pct}%" if fund.management_fee_pct else "      Management Fee: N/A")
            print(f"      Min Investment: R$ {fund.min_initial_investment:,.2f}" if fund.min_initial_investment else "      Min Investment: N/A")
            if fund.searchable_text:
                print(f"      Searchable Text Preview: {fund.searchable_text[:150]}...")
    else:
        print(f"Error: {result.error_message}")

    # Test 3: Multiple CNPJ lookup
    if len(sample) >= 2:
        print("\nTest 3: Multiple CNPJ lookup...")
        test_cnpjs = sample['cnpj'].tolist()[:2]
        print(f"Looking up {len(test_cnpjs)} CNPJs:")
        for cnpj in test_cnpjs:
            print(f"   - {cnpj}")

        result = tool.lookup_by_cnpj(test_cnpjs)

        if result.success:
            print(f"Success! Found {result.total_count} fund(s) in {result.execution_time_ms:.2f}ms")
            for idx, fund in enumerate(result.funds, 1):
                print(f"\n   {idx}. {fund.legal_name} ({fund.cnpj})")
                print(f"      Type: {fund.fund_type}")
                print(f"      NAV: R$ {fund.net_asset_value:,.2f}" if fund.net_asset_value else "      NAV: N/A")
        else:
            print(f"Error: {result.error_message}")

    # Test 4: Non-existent CNPJ
    print("\nTest 4: Non-existent CNPJ lookup...")
    fake_cnpj = "00.000.000/0001-00"
    print(f"Looking up: {fake_cnpj}")

    result = tool.lookup_by_cnpj(fake_cnpj)

    if result.success:
        print(f"Query successful (but no results expected)")
        print(f"   Found: {result.total_count} funds")
        print(f"   Not found: {result.not_found_cnpjs}")
    else:
        print(f"Error: {result.error_message}")

    print("\n" + "=" * 80)
    print("All tests completed!\n")


if __name__ == "__main__":
    test_cnpj_tool()
