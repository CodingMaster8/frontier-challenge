"""Test cases for Agent Routing evaluation."""

from ..models import EvalCase, ToolType

# Agent Routing Test Cases - Testing the agent's ability to choose the right tool
AGENT_ROUTING_CASES = [
    # Should route to semantic search
    EvalCase(
        id="route_sem_001",
        name="Route to semantic - fuzzy search",
        description="Query should be routed to semantic search",
        tool_type=ToolType.AGENT_ROUTING,
        input_query="Find me sustainable investment funds",
        tags=["routing", "to_semantic"],
        metadata={"expected_tool": "semantic_search"},
    ),
    EvalCase(
        id="route_sem_002",
        name="Route to semantic - company name",
        description="Query with company name should use semantic",
        tool_type=ToolType.AGENT_ROUTING,
        input_query="Bradesco investment funds",
        tags=["routing", "to_semantic"],
        metadata={"expected_tool": "semantic_search"},
    ),

    # Should route to structured filter
    EvalCase(
        id="route_sql_001",
        name="Route to SQL - numeric filter",
        description="Query with numbers should route to SQL",
        tool_type=ToolType.AGENT_ROUTING,
        input_query="Funds with returns above 15%",
        tags=["routing", "to_sql"],
        metadata={"expected_tool": "structured_filter"},
    ),
    EvalCase(
        id="route_sql_002",
        name="Route to SQL - fee comparison",
        description="Query about fees should use SQL",
        tool_type=ToolType.AGENT_ROUTING,
        input_query="Show me funds with management fees below 1.5%",
        tags=["routing", "to_sql"],
        metadata={"expected_tool": "structured_filter"},
    ),
    EvalCase(
        id="route_sql_003",
        name="Route to SQL - multiple criteria",
        description="Complex filter criteria should use SQL",
        tool_type=ToolType.AGENT_ROUTING,
        input_query="Large funds with good returns and low fees",
        tags=["routing", "to_sql"],
        metadata={"expected_tool": "structured_filter"},
    ),

    # Should route to holdings search
    EvalCase(
        id="route_hold_001",
        name="Route to holdings - company holdings",
        description="Query about specific holdings should route to holdings tool",
        tool_type=ToolType.AGENT_ROUTING,
        input_query="Which funds hold Petrobras?",
        tags=["routing", "to_holdings"],
        metadata={"expected_tool": "holdings_search"},
    ),
    EvalCase(
        id="route_hold_002",
        name="Route to holdings - exposure query",
        description="Exposure query should use holdings tool",
        tool_type=ToolType.AGENT_ROUTING,
        input_query="Funds with exposure to Vale",
        tags=["routing", "to_holdings"],
        metadata={"expected_tool": "holdings_search"},
    ),
    EvalCase(
        id="route_hold_003",
        name="Route to holdings - portfolio composition",
        description="Portfolio composition query",
        tool_type=ToolType.AGENT_ROUTING,
        input_query="What companies are in this fund's portfolio?",
        tags=["routing", "to_holdings"],
        metadata={"expected_tool": "holdings_search"},
    ),

    # Should route to visualization
    EvalCase(
        id="route_viz_001",
        name="Route to viz - chart request",
        description="Explicit chart request should route to viz",
        tool_type=ToolType.AGENT_ROUTING,
        input_query="Create a chart showing top funds by return",
        tags=["routing", "to_viz"],
        metadata={"expected_tool": "visualization"},
    ),
    EvalCase(
        id="route_viz_002",
        name="Route to viz - comparison request",
        description="Comparison visualization request",
        tool_type=ToolType.AGENT_ROUTING,
        input_query="Compare the performance of these funds",
        tags=["routing", "to_viz"],
        metadata={"expected_tool": "visualization"},
    ),

    # Ambiguous routing cases
    EvalCase(
        id="route_ambig_001",
        name="Ambiguous - could be semantic or SQL",
        description="Query that could use either tool",
        tool_type=ToolType.AGENT_ROUTING,
        input_query="Good performing funds",
        tags=["routing", "ambiguous"],
        edge_case=True,
    ),
    EvalCase(
        id="route_ambig_002",
        name="Ambiguous - mixed criteria",
        description="Query with both semantic and structured elements",
        tool_type=ToolType.AGENT_ROUTING,
        input_query="Conservative Bradesco funds with low fees",
        tags=["routing", "ambiguous"],
        edge_case=True,
    ),

    # Multi-tool scenarios
    EvalCase(
        id="route_multi_001",
        name="Multi-tool - search then filter",
        description="Query requiring multiple tools",
        tool_type=ToolType.AGENT_ROUTING,
        input_query="Find Itau funds and show me which ones have returns above 10%",
        tags=["routing", "multi_tool"],
        edge_case=True,
        metadata={"expected_tools": ["semantic_search", "structured_filter"]},
    ),
    EvalCase(
        id="route_multi_002",
        name="Multi-tool - holdings then viz",
        description="Query requiring holdings search and visualization",
        tool_type=ToolType.AGENT_ROUTING,
        input_query="Show me a chart of funds holding Petrobras by portfolio weight",
        tags=["routing", "multi_tool"],
        edge_case=True,
        metadata={"expected_tools": ["holdings_search", "visualization"]},
    ),
]


def get_agent_routing_suite():
    """Get the agent routing evaluation suite."""
    from ..evaluator import EvalSuite

    return EvalSuite(
        name="Agent Routing Evaluation",
        description="Evaluation of the agent's ability to route queries to the correct tool(s)",
        cases=AGENT_ROUTING_CASES,
        tool_type=ToolType.AGENT_ROUTING,
    )
