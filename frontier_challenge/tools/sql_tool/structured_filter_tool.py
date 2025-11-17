"""
Fund Structured Filter Tool: Production-grade Text-to-SQL for fund filtering

This tool enables natural language queries to be converted to SQL queries against
the fund_structured_filter_view. It uses LangChain/LangGraph for robust query
generation with validation and error recovery.

Key Features:
- Text-to-SQL conversion using OpenAI models
- Query validation and syntax checking
- Automatic retry on errors
- Type-safe results with Pydantic models

Example queries:
- "Funds with >15% YTD return and <2% fees"
- "Large cap equity funds with >R$100M AUM"
- "Funds with low volatility and positive Sharpe ratio"
- "Multimercado funds that beat their benchmark"
"""

import logging
import os
from datetime import datetime
from typing import List, Optional
import asyncio

import duckdb
import pandas as pd

from .models import FundFilterCriteria, FundRecord, FilterResult, FilterQueryState
from .graph import get_graph

logger = logging.getLogger(__name__)


# ============================================================================
# Main Tool Class
# ============================================================================


class StructuredFilterTool:
    """
    Production-grade structured filter tool with text-to-SQL capabilities.

    This tool converts natural language queries into SQL queries against the
    fund_structured_filter_view, with validation, error recovery, and retry logic.
    """

    def __init__(
        self,
        db_path: str = "data/br_funds.db",
        max_retries: int = 3,
        refine_query: bool = True,
    ):
        """
        Initialize the structured filter tool.

        Parameters
        ----------
        db_path : str
            Path to DuckDB database
        max_retries : int
            Maximum retries on query errors
        """
        self.db_path = db_path
        self.max_retries = max_retries
        self.refine_query = refine_query

        # Load view schema
        self.view_schema = self._load_view_schema()

        logger.info(f"Initialized StructuredFilterTool with db: {db_path}")

    def _load_view_schema(self) -> str:
        """Load the schema of fund_structured_filter_view"""
        try:
            conn = duckdb.connect(self.db_path, read_only=True)

            # Get column info
            result = conn.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'fund_structured_filter_view'
                ORDER BY ordinal_position
            """).fetchall()

            conn.close()

            if not result:
                raise ValueError("fund_structured_filter_view not found in database")

            schema_lines = ["fund_structured_filter_view columns:"]
            for col_name, col_type in result:
                schema_lines.append(f"  - {col_name}: {col_type}")

            return "\n".join(schema_lines)

        except Exception as e:
            logger.error(f"Error loading view schema: {e}")
            return "Schema information not available"

    async def structured_filter(
        self,
        query: Optional[str] = None,
        criteria: Optional[FundFilterCriteria] = None,
    ) -> FilterResult:
        """
        Filter funds using natural language query, structured criteria, or template.

        Parameters
        ----------
        query : str, optional
            Natural language query (e.g., "Funds with >15% YTD return and <2% fees")
        criteria : FundFilterCriteria, optional
            Structured filter criteria

        Returns
        -------
        FilterResult
            Results with funds list, SQL query, and metadata
        """
        start_time = datetime.now()

        try:
            # Determine SQL query source
            if criteria:
                sql_query = self._criteria_to_sql(criteria)
            elif query:
                sql_query = await self._text_to_sql(query)
            else:
                return FilterResult(
                    success=False,
                    error_message="Must provide query, criteria, or template"
                )

            # Execute query
            df = self._execute_query(sql_query)

            # Convert to fund records
            funds = self._df_to_funds(df)

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return FilterResult(
                success=True,
                funds=funds,
                total_count=len(funds),
                sql_query=sql_query,
                execution_time_ms=execution_time
            )

        except Exception as e:
            logger.error(f"Error filtering funds: {e}")
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return FilterResult(
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time
            )

    def _criteria_to_sql(self, criteria: FundFilterCriteria) -> str:
        """Convert structured criteria to SQL query"""
        conditions = []

        # Performance filters
        if criteria.min_return_ytd is not None:
            conditions.append(f"return_ytd_2024_avg > {criteria.min_return_ytd}")
        if criteria.min_return_12m is not None:
            conditions.append(f"return_12m_avg > {criteria.min_return_12m}")
        if criteria.min_return_5y is not None:
            conditions.append(f"return_5y_pct > {criteria.min_return_5y}")
        if criteria.max_volatility is not None:
            conditions.append(f"volatility_12m < {criteria.max_volatility}")
        if criteria.min_sharpe_ratio is not None:
            conditions.append(f"sharpe_ratio_approx > {criteria.min_sharpe_ratio}")

        # Fee filters
        if criteria.max_management_fee is not None:
            conditions.append(f"management_fee_pct < {criteria.max_management_fee}")
        if criteria.max_performance_fee is not None:
            conditions.append(f"performance_fee_pct < {criteria.max_performance_fee}")
        if criteria.max_expense_ratio is not None:
            conditions.append(f"expense_ratio_pct < {criteria.max_expense_ratio}")

        # Size filters
        if criteria.min_nav is not None:
            conditions.append(f"nav > {criteria.min_nav}")
        if criteria.max_nav is not None:
            conditions.append(f"nav < {criteria.max_nav}")

        # Investment constraints
        if criteria.max_min_investment is not None:
            conditions.append(f"min_initial_investment <= {criteria.max_min_investment}")
        if criteria.max_lockup_days is not None:
            conditions.append(f"lockup_days <= {criteria.max_lockup_days}")

        # Classification filters
        if criteria.investment_class:
            classes = "', '".join(criteria.investment_class)
            conditions.append(f"investment_class IN ('{classes}')")
        if criteria.fund_type:
            types = "', '".join(criteria.fund_type)
            conditions.append(f"fund_type IN ('{types}')")
        if criteria.benchmark:
            conditions.append(f"performance_benchmark LIKE '%{criteria.benchmark}%'")
        if criteria.risk_class:
            classes = "', '".join(criteria.risk_class)
            conditions.append(f"risk_class IN ('{classes}')")

        # Boolean filters
        if criteria.is_fund_of_funds is not None:
            conditions.append(f"is_fund_of_funds = {criteria.is_fund_of_funds}")
        if criteria.is_exclusive is not None:
            conditions.append(f"is_exclusive_fund = {criteria.is_exclusive}")
        if criteria.can_invest_abroad is not None:
            conditions.append(f"can_invest_abroad_100_pct = {criteria.can_invest_abroad}")

        # Build SQL
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        order_direction = "DESC" if criteria.sort_descending else "ASC"

        sql = f"""
        SELECT *
        FROM fund_structured_filter_view
        WHERE {where_clause}
        ORDER BY {criteria.sort_by} {order_direction}
        LIMIT {criteria.limit}
        """

        return sql

    async def _text_to_sql(self, query: str) -> str:
        """Convert natural language query to SQL using LangGraph workflow"""
        # Build the workflow
        workflow = get_graph(self.db_path, self.max_retries, self.refine_query)

        # Run the workflow
        initial_state = {
            "natural_language_query": query,
            "view_schema": self.view_schema,
        }

        # Use await since we're in async context
        final_state_raw = await workflow.ainvoke(initial_state)

        # Handle both dict and FilterQueryState instances
        # LangGraph may return either depending on version
        if isinstance(final_state_raw, dict):
            # If it's a dict, convert to FilterQueryState for type safety
            final_state: FilterQueryState = FilterQueryState(**final_state_raw)
        else:
            # Already a FilterQueryState instance
            final_state: FilterQueryState = final_state_raw

        if final_state.error_message:
            raise ValueError(f"SQL generation failed: {final_state.error_message}")

        if not final_state.final_sql:
            raise ValueError("No SQL query was generated")

        return final_state.final_sql

    def _execute_query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            df = conn.execute(sql).fetchdf()
            return df
        finally:
            conn.close()

    def _df_to_funds(self, df: pd.DataFrame) -> List[FundRecord]:
        """Convert DataFrame to list of FundRecord objects"""
        # Replace pandas NA with None for proper Pydantic validation
        # Convert to dict with orient='records' which handles NA properly
        records = df.to_dict(orient='records')

        funds = []
        for record in records:
            # Replace pandas NA values with None
            cleaned_record = {k: (None if pd.isna(v) else v) for k, v in record.items()}

            # Convert risk_class to string if present
            if cleaned_record.get("risk_class") is not None:
                cleaned_record["risk_class"] = str(cleaned_record["risk_class"])

            fund = FundRecord(
                fund_id=cleaned_record.get("fund_id", ""),
                cnpj=cleaned_record.get("cnpj", ""),
                legal_name=cleaned_record.get("legal_name", ""),
                trade_name=cleaned_record.get("trade_name"),
                investment_class=cleaned_record.get("investment_class"),
                anbima_classification=cleaned_record.get("anbima_classification"),
                fund_type=cleaned_record.get("fund_type"),
                risk_class=cleaned_record.get("risk_class"),
                nav=cleaned_record.get("nav"),
                management_fee_pct=cleaned_record.get("management_fee_pct"),
                performance_fee_pct=cleaned_record.get("performance_fee_pct"),
                return_ytd_2024_avg=cleaned_record.get("return_ytd_2024_avg"),
                return_12m_avg=cleaned_record.get("return_12m_avg"),
                return_5y_pct=cleaned_record.get("return_5y_pct"),
                volatility_12m=cleaned_record.get("volatility_12m"),
                sharpe_ratio_approx=cleaned_record.get("sharpe_ratio_approx"),
                min_initial_investment=cleaned_record.get("min_initial_investment"),
                lockup_days=cleaned_record.get("lockup_days"),
            )
            funds.append(fund)

        return funds
