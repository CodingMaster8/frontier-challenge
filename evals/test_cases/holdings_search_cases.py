"""Test cases for Holdings Search Tool evaluation."""

from ..models import EvalCase, ToolType

# Holdings Search Test Cases
HOLDINGS_SEARCH_CASES = [
    # Basic company search cases
    EvalCase(
        id="hold_001",
        name="Basic company search - Petrobras",
        description="Search for funds holding Petrobras",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds that invest in Petrobras",
        expected_companies=["PETROBRAS", "PETROBRAS PN", "PETR4"],
        min_results=5,
        tags=["basic", "company_search"],
    ),
    EvalCase(
        id="hold_002",
        name="Basic company search - Vale",
        description="Search for funds holding Vale",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds with Vale holdings",
        expected_companies=["VALE"],
        min_results=5,
        tags=["basic", "company_search"],
    ),
    EvalCase(
        id="hold_003",
        name="Basic company search - Itau",
        description="Search for funds holding Itau",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="which funds own Itau stock",
        expected_companies=["ITAU", "ITAUUNIBANCO"],
        min_results=5,
        tags=["basic", "company_search"],
    ),
    EvalCase(
        id="hold_004",
        name="Bank holdings search",
        description="Search for funds holding major Brazilian banks",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds that invest in Brazilian banks",
        min_results=10,
        tags=["basic", "sector_search"],
    ),
    EvalCase(
        id="hold_005",
        name="Government bonds search",
        description="Search for funds holding government securities",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds with Brazilian government bonds",
        min_results=10,
        tags=["basic", "fixed_income"],
    ),

    # Fuzzy matching cases
    EvalCase(
        id="hold_fuzzy_001",
        name="Fuzzy matching - misspelled company",
        description="Search with misspelled company name",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds holding Petobras",  # Misspelled
        expected_companies=["PETROBRAS"],
        min_results=1,
        tags=["fuzzy", "misspelling"],
    ),
    EvalCase(
        id="hold_fuzzy_002",
        name="Fuzzy matching - partial name",
        description="Search with partial company name",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds with Bradesco",
        min_results=5,
        tags=["fuzzy", "partial_name"],
    ),
    EvalCase(
        id="hold_fuzzy_003",
        name="Fuzzy matching - abbreviated name",
        description="Search with company abbreviation",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds holding BB",  # Banco do Brasil
        min_results=1,
        tags=["fuzzy", "abbreviation"],
    ),

    # Grouping and aggregation
    EvalCase(
        id="hold_group_001",
        name="Group by fund results",
        description="Get results grouped by fund",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds holding Petrobras grouped by fund",
        min_results=5,
        tags=["grouping", "aggregation"],
        metadata={"group_by_fund": True},
    ),
    EvalCase(
        id="hold_group_002",
        name="Top holdings by weight",
        description="Find top positions in a company",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds with largest Vale positions",
        min_results=5,
        tags=["grouping", "ranking"],
    ),

    # Edge cases
    EvalCase(
        id="hold_edge_001",
        name="Non-existent company",
        description="Search for company not in database",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds holding XYZ Company 123",
        max_results=5,
        edge_case=True,
        tags=["edge_case", "not_found"],
    ),
    EvalCase(
        id="hold_edge_002",
        name="Very common term",
        description="Search with very generic term",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds holding bonds",
        min_results=1,
        edge_case=True,
        tags=["edge_case", "generic"],
    ),
    EvalCase(
        id="hold_edge_003",
        name="Multiple company search",
        description="Search for multiple companies at once",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds holding both Petrobras and Vale",
        min_results=1,
        edge_case=True,
        tags=["edge_case", "multiple_companies"],
    ),
    EvalCase(
        id="hold_edge_004",
        name="Foreign company search",
        description="Search for international holdings",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds holding Apple stock",
        min_results=0,  # May or may not have international holdings
        edge_case=True,
        tags=["edge_case", "international"],
    ),
    EvalCase(
        id="hold_edge_005",
        name="Empty/minimal query",
        description="Very short search query",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="Vale",
        min_results=1,
        edge_case=True,
        tags=["edge_case", "short_query"],
    ),
    EvalCase(
        id="hold_edge_006",
        name="Special characters in company name",
        description="Company name with special characters",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds holding companies with S.A.",
        min_results=1,
        edge_case=True,
        tags=["edge_case", "special_chars"],
    ),
    EvalCase(
        id="hold_edge_007",
        name="Case sensitivity test",
        description="Test case-insensitive matching",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds holding PETROBRAS in lowercase: petrobras",
        min_results=1,
        edge_case=True,
        tags=["edge_case", "case_sensitivity"],
    ),
    EvalCase(
        id="hold_edge_008",
        name="Similarity threshold test",
        description="Test with low similarity threshold",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds holding Petro",  # Very partial
        min_results=1,
        edge_case=True,
        tags=["edge_case", "similarity_threshold"],
    ),

    # Natural language understanding
    EvalCase(
        id="hold_nl_001",
        name="Natural language - exposure query",
        description="Natural language query about exposure",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="What funds have exposure to Bradesco?",
        min_results=1,
        tags=["natural_language", "exposure"],
    ),
    EvalCase(
        id="hold_nl_002",
        name="Natural language - Portuguese query",
        description="Query in Portuguese",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="Quais fundos investem em Petrobras?",
        min_results=5,
        tags=["natural_language", "portuguese"],
    ),
    EvalCase(
        id="hold_nl_003",
        name="Natural language - complex query",
        description="Complex natural language query",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="Show me all funds that have positions in Vale with portfolio weight above 5%",
        min_results=1,
        tags=["natural_language", "complex"],
    ),

    # Portfolio weight filtering
    EvalCase(
        id="hold_weight_001",
        name="Minimum weight filter",
        description="Filter holdings by minimum portfolio weight",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds with Petrobras holdings above 3% of portfolio",
        min_results=1,
        tags=["filtering", "portfolio_weight"],
        metadata={"min_weight": 3.0},
    ),
    EvalCase(
        id="hold_weight_002",
        name="Large position search",
        description="Find funds with large positions",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds where Vale is a top 5 holding",
        min_results=1,
        tags=["filtering", "large_positions"],
    ),

    # Sector and industry analysis
    EvalCase(
        id="hold_sector_001",
        name="Sector exposure - Energy",
        description="Find funds with energy sector exposure",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds investing in energy companies like Petrobras",
        min_results=5,
        tags=["sector", "energy"],
    ),
    EvalCase(
        id="hold_sector_002",
        name="Sector exposure - Financial",
        description="Find funds with financial sector exposure",
        tool_type=ToolType.HOLDINGS_SEARCH,
        input_query="funds with banking sector holdings",
        min_results=10,
        tags=["sector", "financial"],
    ),
]


def get_holdings_search_suite():
    """Get the complete holdings search evaluation suite."""
    from ..evaluator import EvalSuite

    return EvalSuite(
        name="Holdings Search Tool Evaluation",
        description="Comprehensive evaluation of the holdings search tool with fuzzy matching using Levenshtein distance",
        cases=HOLDINGS_SEARCH_CASES,
        tool_type=ToolType.HOLDINGS_SEARCH,
    )
