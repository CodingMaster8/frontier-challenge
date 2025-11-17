"""
Data models for CNPJ lookup tool.
"""

from typing import List, Optional
from pydantic import BaseModel, Field


class FundInfo(BaseModel):
    """Fund information retrieved by CNPJ"""

    cnpj: str = Field(description="CNPJ identifier of the fund")
    legal_name: str = Field(description="Legal name of the fund")
    fund_type: Optional[str] = Field(default=None, description="Type of fund")
    searchable_text: Optional[str] = Field(default=None, description="Full searchable text with fund details")
    net_asset_value: Optional[float] = Field(default=None, description="Net asset value in BRL")
    management_fee_pct: Optional[float] = Field(default=None, description="Management fee percentage")
    min_initial_investment: Optional[float] = Field(default=None, description="Minimum initial investment in BRL")


class CNPJLookupResult(BaseModel):
    """Result of CNPJ lookup operation"""

    success: bool = Field(description="Whether the lookup was successful")
    funds: List[FundInfo] = Field(default_factory=list, description="List of fund information")
    total_count: int = Field(default=0, description="Total number of funds found")
    not_found_cnpjs: List[str] = Field(default_factory=list, description="CNPJs that were not found")
    error_message: Optional[str] = Field(default=None, description="Error message if lookup failed")
    execution_time_ms: float = Field(default=0.0, description="Execution time in milliseconds")
