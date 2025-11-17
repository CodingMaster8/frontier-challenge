"""
Holdings Search Tool: Find funds that invest in specific companies or assets.

This tool enables natural language queries to find funds holding specific assets,
with support for fuzzy matching using Levenshtein distance in DuckDB.

Key Features:
- Text-to-SQL conversion for holdings queries
- Fuzzy matching with Levenshtein distance
- Asset/company name search
- Portfolio weight filtering
- Results grouped by fund or detailed holdings

Example queries:
- "Funds that invest in Petrobras"
- "Funds holding Apple stock"
- "Funds with exposure to Vale"
- "Funds investing in Brazilian government bonds"
"""

import logging
from datetime import datetime
from typing import List, Optional

import duckdb
import pandas as pd
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

from .models import (
    HoldingsSearchCriteria,
    HoldingRecord,
    FundSummaryWithHolding,
    HoldingsSearchResult,
    EntityExtractionResult,
)
from .holdings_prompt import ENTITY_EXTRACTION_PROMPT

logger = logging.getLogger(__name__)


class HoldingsSearchTool:
    """
    Holdings search tool with fuzzy matching capabilities.

    This tool searches for funds holding specific assets or companies,
    leveraging DuckDB's Levenshtein distance function for fuzzy matching.
    """

    def __init__(
        self,
        db_path: str = "data/br_funds.db",
        default_similarity_threshold: float = 0.6,
        entity_extraction_model: str = "gpt-4o-mini",
    ):
        """
        Initialize the holdings search tool.

        Parameters
        ----------
        db_path : str
            Path to DuckDB database
        default_similarity_threshold : float
            Default minimum similarity score for fuzzy matching (0-1)
        entity_extraction_model : str
            LLM model for entity extraction (default: gpt-4o-mini)
        """
        self.db_path = db_path
        self.default_similarity_threshold = default_similarity_threshold

        # Load view schema
        self.view_schema = self._load_view_schema()

        logger.info(f"Initialized HoldingsSearchTool with db: {db_path}, entity model: {entity_extraction_model}")

    def _load_view_schema(self) -> str:
        """Load the schema of fund_holdings_detail_view"""
        try:
            conn = duckdb.connect(self.db_path, read_only=True)

            # Get column info
            result = conn.execute("""
                SELECT column_name, data_type
                FROM information_schema.columns
                WHERE table_name = 'fund_holdings_detail_view'
                ORDER BY ordinal_position
            """).fetchall()

            conn.close()

            if not result:
                raise ValueError("fund_holdings_detail_view not found in database")

            schema_lines = ["fund_holdings_detail_view columns:"]
            for col_name, col_type in result:
                schema_lines.append(f"  - {col_name}: {col_type}")

            return "\n".join(schema_lines)

        except Exception as e:
            logger.error(f"Error loading view schema: {e}")
            return "Schema information not available"

    async def _extract_entities_from_query(self, query: str) -> EntityExtractionResult:
        """
        Extract company, bank, and asset names from a natural language query using LLM.

        Parameters
        ----------
        query : str
            Natural language query from user

        Returns
        -------
        EntityExtractionResult
            Extracted entities
        """
        logger.info(f"Extracting entities from query: {query[:100]}...")

        # Initialize LLM for entity extraction
        llm_light = ChatOpenAI(
            model="gpt-5-chat-latest",
            temperature=0,
        ).with_retry()

        try:
            # Build prompt
            prompt_template = ChatPromptTemplate.from_template(ENTITY_EXTRACTION_PROMPT)
            parser = PydanticOutputParser(pydantic_object=EntityExtractionResult)

            # Build chain
            chain = prompt_template | llm_light | parser

            # Extract entities
            result = await chain.ainvoke({
                "query": query,
                "format_instructions": parser.get_format_instructions()
            })

            logger.info(f"Extracted {len(result.entities)} entities: {result.entities}")

            return result

        except Exception as e:
            logger.error(f"Error extracting entities: {e}", exc_info=True)
            # Fallback to empty list
            return EntityExtractionResult(entities=[])

    async def search_holdings(
        self,
        query: Optional[str] = None,
        criteria: Optional[HoldingsSearchCriteria] = None,
        company_name: Optional[str] = None,
    ) -> HoldingsSearchResult:
        """
        Search for funds holding specific assets or companies.

        Parameters
        ----------
        query : str, optional
            Natural language query (e.g., "Funds that invest in Petrobras")
        criteria : HoldingsSearchCriteria, optional
            Structured search criteria
        company_name : str, optional
            Direct company/asset name search (shortcut)

        Returns
        -------
        HoldingsSearchResult
            Results with holdings list, fund summaries, and metadata
        """
        start_time = datetime.now()

        try:
            # If company_name is provided as shortcut, create criteria
            if company_name and not criteria and not query:
                criteria = HoldingsSearchCriteria(company_name=company_name)

            # Determine search method
            if criteria:
                sql_query, search_method = self._criteria_to_sql(criteria)
            elif query:
                # Use LLM to extract entities from the query
                logger.info("Using LLM entity extraction for holdings search")
                extraction_result = await self._extract_entities_from_query(query)

                if extraction_result.entities:
                    # Build SQL with multiple entity searches combined with OR
                    sql_query, search_method = self._entities_to_sql(
                        extraction_result.entities,
                        min_similarity=0.6,
                        group_by_fund=True,
                        limit=20
                    )
                    logger.info(f"Generated SQL for {len(extraction_result.entities)} entities")
                else:
                    # Fallback to simple text-to-SQL if no entities extracted
                    logger.warning("No entities extracted, falling back to simple text-to-SQL")
                    sql_query, search_method = self._simple_text_to_sql(query)
            else:
                return HoldingsSearchResult(
                    success=False,
                    error_message="Must provide query, criteria, or company_name"
                )

            # Execute query
            df = self._execute_query(sql_query)

            # Convert to records
            if criteria and criteria.group_by_fund:
                fund_summaries = self._df_to_fund_summaries(df)
                holdings = []
                unique_funds = len(fund_summaries)
            else:
                holdings = self._df_to_holdings(df)
                fund_summaries = []
                unique_funds = df['fund_id'].nunique() if len(df) > 0 else 0

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return HoldingsSearchResult(
                success=True,
                holdings=holdings,
                fund_summaries=fund_summaries,
                total_count=len(holdings) + len(fund_summaries),
                unique_funds_count=unique_funds,
                sql_query=sql_query,
                search_method=search_method,
                execution_time_ms=execution_time
            )

        except Exception as e:
            logger.error(f"Error searching holdings: {e}")
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return HoldingsSearchResult(
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time
            )

    def _criteria_to_sql(self, criteria: HoldingsSearchCriteria) -> tuple[str, str]:
        """
        Convert structured criteria to SQL query with fuzzy matching support.

        Returns tuple of (sql_query, search_method)
        """
        conditions = []
        search_method = "exact"

        # Company/Asset name search with fuzzy matching
        if criteria.use_fuzzy_match and (criteria.company_name or criteria.issuer_name or criteria.asset_name):
            search_method = "fuzzy"
            search_term = criteria.company_name or criteria.issuer_name or criteria.asset_name
            min_similarity = criteria.min_similarity

            # DuckDB Levenshtein distance fuzzy matching
            # Calculate similarity score: 1 - (levenshtein_distance / max_length)
            fuzzy_conditions = []

            if criteria.company_name or criteria.asset_name:
                fuzzy_conditions.append(f"""
                    (1.0 - CAST(levenshtein(LOWER(asset_name), LOWER('{search_term}')) AS DOUBLE) /
                     GREATEST(LENGTH(asset_name), LENGTH('{search_term}'))) >= {min_similarity}
                """)
                fuzzy_conditions.append(f"""
                    (1.0 - CAST(levenshtein(LOWER(asset_short_name), LOWER('{search_term}')) AS DOUBLE) /
                     GREATEST(LENGTH(asset_short_name), LENGTH('{search_term}'))) >= {min_similarity}
                """)

            if criteria.company_name or criteria.issuer_name:
                fuzzy_conditions.append(f"""
                    (1.0 - CAST(levenshtein(LOWER(issuer_name), LOWER('{search_term}')) AS DOUBLE) /
                     GREATEST(LENGTH(issuer_name), LENGTH('{search_term}'))) >= {min_similarity}
                """)

            if fuzzy_conditions:
                conditions.append(f"({' OR '.join(fuzzy_conditions)})")

        else:
            # Exact matching with LIKE
            if criteria.company_name:
                conditions.append(f"""
                    (LOWER(asset_name) LIKE LOWER('%{criteria.company_name}%')
                     OR LOWER(asset_short_name) LIKE LOWER('%{criteria.company_name}%')
                     OR LOWER(issuer_name) LIKE LOWER('%{criteria.company_name}%'))
                """)
            elif criteria.asset_name:
                conditions.append(f"""
                    (LOWER(asset_name) LIKE LOWER('%{criteria.asset_name}%')
                     OR LOWER(asset_short_name) LIKE LOWER('%{criteria.asset_name}%'))
                """)
            elif criteria.issuer_name:
                conditions.append(f"LOWER(issuer_name) LIKE LOWER('%{criteria.issuer_name}%')")

        # Asset filters
        if criteria.asset_class:
            conditions.append(f"asset_class = '{criteria.asset_class}'")
        if criteria.financial_instrument:
            conditions.append(f"financial_instrument = '{criteria.financial_instrument}'")
        if criteria.asset_country:
            conditions.append(f"asset_country = '{criteria.asset_country}'")
        if criteria.asset_currency:
            conditions.append(f"asset_currency = '{criteria.asset_currency}'")

        # Position filters
        if criteria.min_weight is not None:
            conditions.append(f"portfolio_weight_pct >= {criteria.min_weight}")
        if criteria.min_position_value is not None:
            conditions.append(f"position_value >= {criteria.min_position_value}")

        # Fund filters
        if criteria.fund_investment_class:
            conditions.append(f"investment_class = '{criteria.fund_investment_class}'")

        # Build SQL
        where_clause = " AND ".join(conditions) if conditions else "1=1"
        order_direction = "DESC" if criteria.sort_descending else "ASC"

        if criteria.group_by_fund:
            # Group by fund, return one row per fund with top holding
            sql = f"""
            WITH ranked_holdings AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY fund_id ORDER BY portfolio_weight_pct DESC) as rn
                FROM fund_holdings_detail_view
                WHERE {where_clause}
            )
            SELECT
                fund_id,
                cnpj,
                legal_name,
                investment_class,
                asset_name,
                asset_short_name,
                issuer_name,
                portfolio_weight_pct,
                position_value
            FROM ranked_holdings
            WHERE rn = 1
            ORDER BY {criteria.sort_by} {order_direction}
            LIMIT {criteria.limit}
            """
        else:
            # Return all matching holdings
            sql = f"""
            SELECT *
            FROM fund_holdings_detail_view
            WHERE {where_clause}
            ORDER BY {criteria.sort_by} {order_direction}
            LIMIT {criteria.limit}
            """

        return sql, search_method

    def _entities_to_sql(
        self,
        entities: List[str],
        min_similarity: float = 0.6,
        group_by_fund: bool = True,
        limit: int = 50
    ) -> tuple[str, str]:
        """
        Convert extracted entities to SQL query with fuzzy matching.

        This method creates an SQL query that searches for funds holding any of the extracted entities,
        using Levenshtein distance for fuzzy matching on each entity.

        Parameters
        ----------
        entities : List[str]
            List of company/asset names extracted by LLM
        min_similarity : float
            Minimum similarity score for fuzzy matching (0-1)
        group_by_fund : bool
            Whether to group results by fund
        limit : int
            Maximum number of results

        Returns
        -------
        tuple[str, str]
            (sql_query, search_method)
        """
        if not entities:
            # Fallback to match all
            where_clause = "1=1"
        else:
            # Build OR conditions for each entity
            entity_conditions = []
            for entity in entities:
                # Escape single quotes in entity name
                safe_entity = entity.replace("'", "''")

                # Create fuzzy matching conditions for this entity
                fuzzy_conditions = []

                # Match against asset_name
                fuzzy_conditions.append(f"""
                    (1.0 - CAST(levenshtein(LOWER(asset_name), LOWER('{safe_entity}')) AS DOUBLE) /
                     GREATEST(LENGTH(asset_name), LENGTH('{safe_entity}'))) >= {min_similarity}
                """)

                # Match against asset_short_name
                fuzzy_conditions.append(f"""
                    (1.0 - CAST(levenshtein(LOWER(asset_short_name), LOWER('{safe_entity}')) AS DOUBLE) /
                     GREATEST(LENGTH(asset_short_name), LENGTH('{safe_entity}'))) >= {min_similarity}
                """)

                # Match against issuer_name
                fuzzy_conditions.append(f"""
                    (1.0 - CAST(levenshtein(LOWER(issuer_name), LOWER('{safe_entity}')) AS DOUBLE) /
                     GREATEST(LENGTH(issuer_name), LENGTH('{safe_entity}'))) >= {min_similarity}
                """)

                # Combine conditions for this entity with OR
                entity_condition = f"({' OR '.join(fuzzy_conditions)})"
                entity_conditions.append(entity_condition)

            # Combine all entity conditions with OR (match ANY entity)
            where_clause = f"({' OR '.join(entity_conditions)})"

        # Build final SQL
        if group_by_fund:
            sql = f"""
            WITH ranked_holdings AS (
                SELECT *,
                    ROW_NUMBER() OVER (PARTITION BY fund_id ORDER BY portfolio_weight_pct DESC) as rn
                FROM fund_holdings_detail_view
                WHERE {where_clause}
            )
            SELECT
                fund_id,
                cnpj,
                legal_name,
                investment_class,
                asset_name,
                asset_short_name,
                issuer_name,
                issuer_type,
                portfolio_weight_pct,
                position_value,
                asset_country,
                asset_currency
            FROM ranked_holdings
            WHERE rn = 1
            ORDER BY portfolio_weight_pct DESC
            LIMIT {limit}
            """
        else:
            sql = f"""
            SELECT *
            FROM fund_holdings_detail_view
            WHERE {where_clause}
            ORDER BY portfolio_weight_pct DESC
            LIMIT {limit}
            """

        return sql, "fuzzy_llm"

    def _simple_text_to_sql(self, query: str) -> tuple[str, str]:
        """
        Simple text-to-SQL conversion for common queries.
        Extract company names and generate appropriate SQL.

        Returns tuple of (sql_query, search_method)
        """
        # Simple pattern matching to extract company names
        query_lower = query.lower()

        # Common patterns
        patterns = [
            "invest in ",
            "holding ",
            "holds ",
            "exposure to ",
            "invested in ",
            "with ",
        ]

        company_name = None
        for pattern in patterns:
            if pattern in query_lower:
                # Extract text after pattern
                start_idx = query_lower.index(pattern) + len(pattern)
                remaining = query[start_idx:].strip()
                # Take until punctuation or end
                company_name = remaining.split()[0] if remaining else None
                break

        if not company_name:
            # Just use fuzzy match on the whole query
            company_name = query.strip()

        # Create criteria and convert to SQL
        criteria = HoldingsSearchCriteria(
            company_name=company_name,
            use_fuzzy_match=True,
            min_similarity=0.6,
            group_by_fund=True,
            limit=50
        )

        return self._criteria_to_sql(criteria)

    def _execute_query(self, sql: str) -> pd.DataFrame:
        """Execute SQL query and return DataFrame"""
        conn = duckdb.connect(self.db_path, read_only=True)
        try:
            df = conn.execute(sql).fetchdf()
            return df
        finally:
            conn.close()

    def _df_to_holdings(self, df: pd.DataFrame) -> List[HoldingRecord]:
        """Convert DataFrame to list of HoldingRecord objects"""
        records = df.to_dict(orient='records')

        holdings = []
        for record in records:
            # Replace pandas NA values with None
            cleaned_record = {k: (None if pd.isna(v) else v) for k, v in record.items()}

            holding = HoldingRecord(
                fund_id=cleaned_record.get("fund_id", ""),
                cnpj=cleaned_record.get("cnpj", ""),
                legal_name=cleaned_record.get("legal_name", ""),
                investment_class=cleaned_record.get("investment_class"),
                position_id=cleaned_record.get("position_id", ""),
                asset_id=cleaned_record.get("asset_id", ""),
                quantity=cleaned_record.get("quantity"),
                position_value=cleaned_record.get("position_value"),
                position_currency=cleaned_record.get("position_currency"),
                portfolio_weight_pct=cleaned_record.get("portfolio_weight_pct"),
                asset_class=cleaned_record.get("asset_class"),
                financial_instrument=cleaned_record.get("financial_instrument"),
                financial_instrument_description=cleaned_record.get("financial_instrument_description"),
                asset_name=cleaned_record.get("asset_name"),
                asset_short_name=cleaned_record.get("asset_short_name"),
                asset_currency=cleaned_record.get("asset_currency"),
                asset_country=cleaned_record.get("asset_country"),
                issuer_name=cleaned_record.get("issuer_name"),
                issuer_type=cleaned_record.get("issuer_type"),
                position_date=str(cleaned_record.get("position_date")) if cleaned_record.get("position_date") else None,
            )
            holdings.append(holding)

        return holdings

    def _df_to_fund_summaries(self, df: pd.DataFrame) -> List[FundSummaryWithHolding]:
        """Convert DataFrame to list of FundSummaryWithHolding objects"""
        records = df.to_dict(orient='records')

        summaries = []
        for record in records:
            # Replace pandas NA values with None
            cleaned_record = {k: (None if pd.isna(v) else v) for k, v in record.items()}

            summary = FundSummaryWithHolding(
                fund_id=cleaned_record.get("fund_id", ""),
                cnpj=cleaned_record.get("cnpj", ""),
                legal_name=cleaned_record.get("legal_name", ""),
                investment_class=cleaned_record.get("investment_class"),
                asset_name=cleaned_record.get("asset_name", ""),
                asset_short_name=cleaned_record.get("asset_short_name"),
                issuer_name=cleaned_record.get("issuer_name"),
                portfolio_weight_pct=cleaned_record.get("portfolio_weight_pct"),
                position_value=cleaned_record.get("position_value"),
                asset_country=cleaned_record.get("asset_country"),
                asset_currency=cleaned_record.get("asset_currency"),
            )
            summaries.append(summary)

        return summaries
