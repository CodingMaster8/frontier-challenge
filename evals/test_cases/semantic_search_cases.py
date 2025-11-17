"""Test cases for Semantic Search Tool evaluation."""

from ..models import EvalCase, ToolType

# Semantic Search Test Cases
SEMANTIC_SEARCH_CASES = [
    # Basic search cases
    EvalCase(
        id="sem_001",
        name="Basic company name search",
        description="Search for funds by administrator name",
        tool_type=ToolType.SEMANTIC_SEARCH,
        input_query="Bradesco funds",
        min_results=5,
        tags=["basic", "company_name"],
    ),
    EvalCase(
        id="sem_002",
        name="Portuguese investment type search",
        description="Search for conservative fixed income funds in Portuguese",
        tool_type=ToolType.SEMANTIC_SEARCH,
        input_query="fundos de renda fixa conservadores",
        min_results=5,
        tags=["basic", "portuguese", "investment_type"],
    ),
    EvalCase(
        id="sem_003",
        name="ESG/sustainable investing search",
        description="Search for sustainable/ESG funds",
        tool_type=ToolType.SEMANTIC_SEARCH,
        input_query="sustainable ESG investing",
        min_results=1,
        tags=["basic", "esg", "english"],
    ),
    EvalCase(
        id="sem_004",
        name="Fuzzy fund name matching",
        description="Search with partial/fuzzy fund name",
        tool_type=ToolType.SEMANTIC_SEARCH,
        input_query="BB renda fixa",
        min_results=3,
        tags=["fuzzy", "partial_name"],
    ),
    EvalCase(
        id="sem_005",
        name="Risk profile search",
        description="Search by risk characteristics",
        tool_type=ToolType.SEMANTIC_SEARCH,
        input_query="low risk conservative funds",
        min_results=5,
        tags=["basic", "risk_profile"],
    ),

    # Edge cases
    EvalCase(
        id="sem_edge_001",
        name="Very specific niche query",
        description="Search for very specific investment strategy",
        tool_type=ToolType.SEMANTIC_SEARCH,
        input_query="cryptocurrency blockchain technology funds",
        min_results=0,  # May or may not exist
        edge_case=True,
        tags=["edge_case", "niche"],
    ),
    EvalCase(
        id="sem_edge_002",
        name="Misspelled query",
        description="Search with intentional typos",
        tool_type=ToolType.SEMANTIC_SEARCH,
        input_query="Bradsko gold fnd",  # Misspelled Bradesco
        min_results=1,
        edge_case=True,
        tags=["edge_case", "robustness", "misspelling"],
    ),
    EvalCase(
        id="sem_edge_003",
        name="Mixed language query",
        description="Query mixing Portuguese and English",
        tool_type=ToolType.SEMANTIC_SEARCH,
        input_query="fundos de tech companies americanas",
        min_results=1,
        edge_case=True,
        tags=["edge_case", "mixed_language"],
    ),
    EvalCase(
        id="sem_edge_004",
        name="Empty/minimal query",
        description="Very short or empty search query",
        tool_type=ToolType.SEMANTIC_SEARCH,
        input_query="RF",  # Very short
        min_results=1,
        edge_case=True,
        tags=["edge_case", "short_query"],
    ),
    EvalCase(
        id="sem_edge_005",
        name="Complex multi-criteria query",
        description="Query with multiple investment criteria",
        tool_type=ToolType.SEMANTIC_SEARCH,
        input_query="large cap equity funds with low fees and high returns focused on technology sector",
        min_results=1,
        edge_case=True,
        tags=["edge_case", "complex"],
    ),
    EvalCase(
        id="sem_edge_006",
        name="Non-existent fund search",
        description="Search for funds that don't exist",
        tool_type=ToolType.SEMANTIC_SEARCH,
        input_query="XYZ Nonexistent Fund 12345",
        max_results=5,
        edge_case=True,
        tags=["edge_case", "not_found"],
    ),
    EvalCase(
        id="sem_edge_007",
        name="Special characters in query",
        description="Query with special characters",
        tool_type=ToolType.SEMANTIC_SEARCH,
        input_query="Fundos DI @#$% renda fixa",
        min_results=1,
        edge_case=True,
        tags=["edge_case", "special_chars"],
    ),
    EvalCase(
        id="sem_edge_008",
        name="Very long query",
        description="Excessively long query string",
        tool_type=ToolType.SEMANTIC_SEARCH,
        input_query="I am looking for Brazilian investment funds that focus primarily on fixed income securities with a conservative risk profile, preferably managed by well-known institutions like Banco do Brasil or Bradesco, with low management fees and consistent returns over the past 5 years",
        min_results=1,
        edge_case=True,
        tags=["edge_case", "long_query"],
    ),

    # Performance and accuracy cases
    EvalCase(
        id="sem_perf_001",
        name="Top K variation test",
        description="Test with different top_k values",
        tool_type=ToolType.SEMANTIC_SEARCH,
        input_query="Itau investment funds",
        min_results=10,
        max_results=10,
        tags=["performance", "top_k"],
        metadata={"top_k": 10},
    ),
    EvalCase(
        id="sem_acc_001",
        name="Known fund exact match",
        description="Search for a specific known fund",
        tool_type=ToolType.SEMANTIC_SEARCH,
        input_query="BTG Pactual Corporate Fundo de Investimento Multimercado Crédito Privado",
        min_results=1,
        tags=["accuracy", "exact_match"],
    ),
    EvalCase(
        id="sem_acc_002",
        name="Investment class search",
        description="Search by investment classification",
        tool_type=ToolType.SEMANTIC_SEARCH,
        input_query="multimercado macro",
        min_results=5,
        tags=["accuracy", "classification"],
    ),
    EvalCase(
        id="sem_acc_003",
        name="Geographic focus search",
        description="Search for funds with specific geographic focus",
        tool_type=ToolType.SEMANTIC_SEARCH,
        input_query="fundos com foco em mercado brasileiro",
        min_results=5,
        tags=["accuracy", "geographic"],
    ),
]


def get_semantic_search_suite():
    """Get the complete semantic search evaluation suite."""
    from ..evaluator import EvalSuite

    return EvalSuite(
        name="Semantic Search Tool Evaluation",
        description="Comprehensive evaluation of the semantic search tool using Pinecone + OpenAI embeddings",
        cases=SEMANTIC_SEARCH_CASES,
        tool_type=ToolType.SEMANTIC_SEARCH,
    )
