"""
CNPJ Lookup Tool: Retrieve fund information by CNPJ identifier(s)

This tool enables quick lookup of fund information using one or more CNPJ numbers.
It queries the fund_semantic_search_view and returns essential fund details.

Key Features:
- Single or batch CNPJ lookup
- Returns key fund information (name, type, fees, etc.)
- Fast direct database queries
- Type-safe results with Pydantic models

Example usage:
- Get details for CNPJ "12.345.678/0001-90"
- Lookup multiple funds by their CNPJs
- Retrieve fund information for comparison
"""

import logging
import re
from datetime import datetime
from typing import List, Union

import duckdb
import pandas as pd

from .models import FundInfo, CNPJLookupResult

logger = logging.getLogger(__name__)


class CNPJLookupTool:
    """
    CNPJ lookup tool for direct fund information retrieval.

    This tool provides fast lookup of fund information using CNPJ identifiers,
    returning only the most relevant fields for quick reference.
    """

    def __init__(self, db_path: str = "data/br_funds.db"):
        """
        Initialize the CNPJ lookup tool.

        Parameters
        ----------
        db_path : str
            Path to DuckDB database
        """
        self.db_path = db_path
        logger.info(f"Initialized CNPJLookupTool with db: {db_path}")

    def _extract_cnpjs_from_text(self, text: str) -> List[str]:
        """
        Extract CNPJ numbers from text using regex patterns.

        Supports both formatted and unformatted CNPJs:
        - Formatted: 12.345.678/0001-90
        - Unformatted: 12345678000190

        Parameters
        ----------
        text : str
            Text that may contain one or more CNPJ numbers

        Returns
        -------
        List[str]
            List of extracted CNPJ numbers (can be empty if none found)
        """
        cnpjs = []

        # Pattern 1: Formatted CNPJ (XX.XXX.XXX/XXXX-XX)
        formatted_pattern = r'\b\d{2}\.\d{3}\.\d{3}/\d{4}-\d{2}\b'
        formatted_matches = re.findall(formatted_pattern, text)
        cnpjs.extend(formatted_matches)

        # Pattern 2: Unformatted CNPJ (14 consecutive digits)
        # Use negative lookahead/lookbehind to avoid matching longer numbers
        unformatted_pattern = r'(?<!\d)\d{14}(?!\d)'
        unformatted_matches = re.findall(unformatted_pattern, text)
        cnpjs.extend(unformatted_matches)

        # Remove duplicates while preserving order
        seen = set()
        unique_cnpjs = []
        for cnpj in cnpjs:
            # Normalize to digits-only for deduplication
            digits = ''.join(c for c in cnpj if c.isdigit())
            if digits not in seen:
                seen.add(digits)
                unique_cnpjs.append(cnpj)

        logger.debug(f"Extracted {len(unique_cnpjs)} CNPJ(s) from text: {unique_cnpjs}")
        return unique_cnpjs

    def _normalize_cnpj(self, cnpj: str) -> str:
        """
        Normalize CNPJ by removing formatting characters.

        Parameters
        ----------
        cnpj : str
            CNPJ with or without formatting (e.g., "12.345.678/0001-90" or "12345678000190")

        Returns
        -------
        str
            Normalized CNPJ with dots, slashes, and hyphens
        """
        # Remove all non-digit characters
        digits_only = ''.join(c for c in cnpj if c.isdigit())

        # If we have 14 digits, format it properly
        if len(digits_only) == 14:
            # Format as XX.XXX.XXX/XXXX-XX
            return f"{digits_only[:2]}.{digits_only[2:5]}.{digits_only[5:8]}/{digits_only[8:12]}-{digits_only[12:14]}"

        # Return as-is if not 14 digits (let DB handle it)
        return cnpj

    def lookup_by_cnpj(
        self,
        cnpj: Union[str, List[str]],
        auto_extract: bool = True
    ) -> CNPJLookupResult:
        """
        Look up fund information by CNPJ number(s).

        This method intelligently handles both direct CNPJ inputs and text queries
        that contain CNPJ numbers.

        Parameters
        ----------
        cnpj : str or List[str]
            Can be:
            - A single CNPJ: "12.345.678/0001-90" or "12345678000190"
            - A list of CNPJs: ["12.345.678/0001-90", "98765432000100"]
            - A text query containing CNPJs: "give me details of 12.345.678/0001-90 and 98765432000100"
        auto_extract : bool, optional
            If True (default), automatically extract CNPJs from text using regex.
            If False, treat input as literal CNPJ values only.

        Returns
        -------
        CNPJLookupResult
            Results with fund information and metadata
        """
        start_time = datetime.now()

        try:
            # Normalize input to list
            if isinstance(cnpj, str):
                # Try to extract CNPJs from the string if auto_extract is enabled
                if auto_extract:
                    extracted_cnpjs = self._extract_cnpjs_from_text(cnpj)
                    if extracted_cnpjs:
                        logger.info(f"Auto-extracted {len(extracted_cnpjs)} CNPJ(s) from query")
                        cnpj_list = extracted_cnpjs
                    else:
                        # No CNPJs found via regex, treat as literal (might be unformatted)
                        cnpj_list = [cnpj]
                else:
                    cnpj_list = [cnpj]
            else:
                cnpj_list = cnpj

            if not cnpj_list:
                return CNPJLookupResult(
                    success=False,
                    error_message="No CNPJ provided"
                )

            # Normalize all CNPJs
            normalized_cnpjs = [self._normalize_cnpj(c) for c in cnpj_list]

            # Build SQL query
            # Use LIKE to handle both formatted and unformatted CNPJs
            cnpj_conditions = []
            for norm_cnpj in normalized_cnpjs:
                # Extract just digits for flexible matching
                digits = ''.join(c for c in norm_cnpj if c.isdigit())
                cnpj_conditions.append(f"REPLACE(REPLACE(REPLACE(cnpj, '.', ''), '/', ''), '-', '') = '{digits}'")

            where_clause = " OR ".join(cnpj_conditions)

            sql_query = f"""
            SELECT
                cnpj,
                legal_name,
                fund_type,
                searchable_text,
                net_asset_value,
                management_fee_pct,
                min_initial_investment
            FROM fund_semantic_search_view
            WHERE {where_clause}
            """

            logger.info(f"Executing CNPJ lookup for {len(cnpj_list)} CNPJ(s)")

            # Execute query
            conn = duckdb.connect(self.db_path, read_only=True)
            df = conn.execute(sql_query).fetchdf()
            conn.close()

            # Convert to FundInfo objects
            funds = []
            for _, row in df.iterrows():
                fund = FundInfo(
                    cnpj=row['cnpj'],
                    legal_name=row['legal_name'],
                    fund_type=row['fund_type'] if pd.notna(row['fund_type']) else None,
                    searchable_text=row['searchable_text'] if pd.notna(row['searchable_text']) else None,
                    net_asset_value=float(row['net_asset_value']) if pd.notna(row['net_asset_value']) else None,
                    management_fee_pct=float(row['management_fee_pct']) if pd.notna(row['management_fee_pct']) else None,
                    min_initial_investment=float(row['min_initial_investment']) if pd.notna(row['min_initial_investment']) else None,
                )
                funds.append(fund)

            # Find CNPJs that were not found
            found_cnpjs_digits = set(''.join(c for c in f.cnpj if c.isdigit()) for f in funds)
            not_found = [
                cnpj for cnpj in normalized_cnpjs
                if ''.join(c for c in cnpj if c.isdigit()) not in found_cnpjs_digits
            ]

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            logger.info(f"✅ Found {len(funds)} fund(s) in {execution_time:.2f}ms")

            if not_found:
                logger.warning(f"⚠️  {len(not_found)} CNPJ(s) not found: {not_found}")

            return CNPJLookupResult(
                success=True,
                funds=funds,
                total_count=len(funds),
                not_found_cnpjs=not_found,
                execution_time_ms=execution_time
            )

        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            logger.error(f"❌ Error in CNPJ lookup: {e}", exc_info=True)

            return CNPJLookupResult(
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time
            )
