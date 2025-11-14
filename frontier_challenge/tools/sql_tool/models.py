from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator
import pandas as pd

# todo: Add field_valdators where appropriate

# ============================================================================
# Pydantic Models for Type Safety
# ============================================================================

class FundFilterCriteria(BaseModel):
    """Structured filter criteria for funds"""

    # Performance filters
    min_return_ytd: Optional[float] = Field(None, description="Minimum YTD return (%)")
    min_return_12m: Optional[float] = Field(None, description="Minimum 12-month return (%)")
    min_return_5y: Optional[float] = Field(None, description="Minimum 5-year return (%)")
    max_volatility: Optional[float] = Field(None, description="Maximum volatility")
    min_sharpe_ratio: Optional[float] = Field(None, description="Minimum Sharpe ratio")

    # Fee filters
    max_management_fee: Optional[float] = Field(None, description="Maximum management fee (%)")
    max_performance_fee: Optional[float] = Field(None, description="Maximum performance fee (%)")
    max_expense_ratio: Optional[float] = Field(None, description="Maximum expense ratio (%)")

    # Size filters
    min_nav: Optional[float] = Field(None, description="Minimum net asset value (R$)")
    max_nav: Optional[float] = Field(None, description="Maximum net asset value (R$)")

    # Investment constraints
    max_min_investment: Optional[float] = Field(None, description="Maximum minimum initial investment (R$)")
    max_lockup_days: Optional[int] = Field(None, description="Maximum lockup period (days)")

    # Classification filters
    investment_class: Optional[List[str]] = Field(None, description="Investment class(es)")
    fund_type: Optional[List[str]] = Field(None, description="Fund type(s)")
    benchmark: Optional[str] = Field(None, description="Performance benchmark")
    risk_class: Optional[List[str]] = Field(None, description="Risk class(es)")

    # Boolean filters
    is_fund_of_funds: Optional[bool] = Field(None, description="Filter for fund of funds")
    is_exclusive: Optional[bool] = Field(None, description="Filter for exclusive funds")
    can_invest_abroad: Optional[bool] = Field(None, description="Can invest 100% abroad")

    # Sorting
    sort_by: Optional[str] = Field("return_ytd_2024_avg", description="Field to sort by")
    sort_descending: Optional[bool] = Field(True, description="Sort in descending order")

    # Limit
    limit: Optional[int] = Field(50, description="Maximum number of results", ge=1, le=1000)


class FundRecord(BaseModel):
    """A single fund record from the filter view"""

    fund_id: str
    cnpj: str
    legal_name: str
    trade_name: Optional[str] = None

    # Classification
    investment_class: Optional[str] = None
    anbima_classification: Optional[str] = None
    fund_type: Optional[str] = None
    risk_class: Optional[str] = None

    # Size and fees
    nav: Optional[float] = None
    management_fee_pct: Optional[float] = None
    performance_fee_pct: Optional[float] = None

    # Performance
    return_ytd_2024_avg: Optional[float] = None
    return_12m_avg: Optional[float] = None
    return_5y_pct: Optional[float] = None
    volatility_12m: Optional[float] = None
    sharpe_ratio_approx: Optional[float] = None

    # Investment constraints
    min_initial_investment: Optional[float] = None
    lockup_days: Optional[int] = None

    class Config:
        arbitrary_types_allowed = True


class FilterResult(BaseModel):
    """Result from a filter operation"""

    success: bool = True
    funds: List[FundRecord] = Field(default_factory=list)
    total_count: int = 0
    sql_query: Optional[str] = None
    execution_time_ms: Optional[float] = None
    error_message: Optional[str] = None

    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# LangGraph State for Text-to-SQL Workflow
# ============================================================================


class FilterQueryState(BaseModel):
    """State for the filter query workflow"""

    # Input
    natural_language_query: str
    user_criteria: Optional[FundFilterCriteria] = None

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
