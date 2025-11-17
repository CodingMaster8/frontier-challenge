"""
Pydantic models for the Holdings Search Tool.

These models provide type safety for holdings search queries and results.
"""

from typing import Any, List, Optional
from pydantic import BaseModel, Field
import pandas as pd


class HoldingRecord(BaseModel):
    """A single holding record from fund_holdings_detail_view"""

    # Fund information
    fund_id: str
    cnpj: str
    legal_name: str
    investment_class: Optional[str] = None

    # Position information
    position_id: str
    asset_id: str
    quantity: Optional[float] = None
    position_value: Optional[float] = None
    position_currency: Optional[str] = None
    portfolio_weight_pct: Optional[float] = None

    # Asset information
    asset_class: Optional[str] = None
    financial_instrument: Optional[str] = None
    financial_instrument_description: Optional[str] = None
    asset_name: Optional[str] = None
    asset_short_name: Optional[str] = None
    asset_currency: Optional[str] = None
    asset_country: Optional[str] = None
    issuer_name: Optional[str] = None
    issuer_type: Optional[str] = None

    # Metadata
    position_date: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class FundSummaryWithHolding(BaseModel):
    """Summary of a fund with a specific holding"""

    fund_id: str
    cnpj: str
    legal_name: str
    investment_class: Optional[str] = None

    # Holding summary
    asset_name: str
    asset_short_name: Optional[str] = None
    issuer_name: Optional[str] = None
    portfolio_weight_pct: Optional[float] = None
    position_value: Optional[float] = None

    # Matching metadata
    match_score: Optional[float] = None  # Levenshtein similarity score
    matched_field: Optional[str] = None  # Which field matched (asset_name, issuer_name, etc.)

    class Config:
        arbitrary_types_allowed = True


class HoldingsSearchCriteria(BaseModel):
    """Structured search criteria for holdings"""

    # Company/Asset search
    company_name: Optional[str] = Field(None, description="Company or asset name to search for")
    issuer_name: Optional[str] = Field(None, description="Issuer name to search for")
    asset_name: Optional[str] = Field(None, description="Asset name to search for")

    # Fuzzy matching parameters
    use_fuzzy_match: bool = Field(True, description="Use Levenshtein distance for fuzzy matching")
    min_similarity: float = Field(0.6, description="Minimum similarity score (0-1) for fuzzy matching", ge=0, le=1)

    # Asset filters
    asset_class: Optional[str] = Field(None, description="Filter by asset class (EQUITY, FIXED_INCOME, etc.)")
    financial_instrument: Optional[str] = Field(None, description="Filter by financial instrument type")
    asset_country: Optional[str] = Field(None, description="Filter by asset country (e.g., 'BRA', 'USA')")
    asset_currency: Optional[str] = Field(None, description="Filter by asset currency (e.g., 'BRL', 'USD')")

    # Position filters
    min_weight: Optional[float] = Field(None, description="Minimum portfolio weight (%)")
    min_position_value: Optional[float] = Field(None, description="Minimum position value (R$)")

    # Fund filters
    fund_investment_class: Optional[str] = Field(None, description="Filter by fund investment class")

    # Sorting and limiting
    sort_by: str = Field("portfolio_weight_pct", description="Field to sort by")
    sort_descending: bool = Field(True, description="Sort in descending order")
    limit: int = Field(100, description="Maximum number of results", ge=1, le=1000)

    # Result grouping
    group_by_fund: bool = Field(False, description="Group results by fund (one row per fund)")


class HoldingsSearchResult(BaseModel):
    """Result from a holdings search operation"""

    success: bool = True
    holdings: List[HoldingRecord] = Field(default_factory=list)
    fund_summaries: List[FundSummaryWithHolding] = Field(default_factory=list)
    total_count: int = 0
    unique_funds_count: int = 0
    sql_query: Optional[str] = None
    search_method: Optional[str] = None  # "exact", "fuzzy", "sql"
    execution_time_ms: Optional[float] = None
    error_message: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


class HoldingsQueryState(BaseModel):
    """State for the holdings search workflow (LangGraph)"""

    # Input
    natural_language_query: str
    user_criteria: Optional[HoldingsSearchCriteria] = None

    # Schema context
    view_schema: Optional[str] = None

    # Generated SQL
    generated_sql: Optional[str] = None
    refined_sql: Optional[str] = None
    final_sql: Optional[str] = None

    # Messages for LLM interactions
    messages: List[Any] = Field(default_factory=list)

    # Execution results
    query_result: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None
    retry_count: int = 0

    # Metadata
    execution_time_ms: Optional[float] = None

    class Config:
        arbitrary_types_allowed = True
