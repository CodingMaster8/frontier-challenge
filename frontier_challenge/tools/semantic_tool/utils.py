"""
Utility functions for Semantic Search Tool

Helper functions for text processing, embedding generation, and result formatting.
"""

import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


def format_fund_text_for_embedding(
    legal_name: str,
    trade_name: Optional[str] = None,
    investment_class: Optional[str] = None,
    anbima_classification: Optional[str] = None,
    objective: Optional[str] = None,
    investment_policy: Optional[str] = None,
) -> str:
    """
    Format fund information into a rich text representation for embedding.

    This function creates a comprehensive text that captures the fund's identity
    and characteristics for semantic search.

    Parameters
    ----------
    legal_name : str
        Fund legal name (required)
    trade_name : str, optional
        Fund trade name
    investment_class : str, optional
        Investment class (e.g., "Ações", "Renda Fixa")
    anbima_classification : str, optional
        ANBIMA classification
    objective : str, optional
        Fund objective/description
    investment_policy : str, optional
        Investment policy details

    Returns
    -------
    str
        Formatted text for embedding
    """
    parts = [legal_name]

    if trade_name:
        parts.append(f"Trade Name: {trade_name}")

    if investment_class:
        parts.append(f"Class: {investment_class}")

    if anbima_classification:
        parts.append(f"Classification: {anbima_classification}")

    if objective:
        parts.append(f"Objective: {objective}")

    if investment_policy:
        parts.append(f"Policy: {investment_policy}")

    return " | ".join(parts)


def calculate_data_quality_score(
    has_objective: bool,
    has_policy: bool,
    has_trade_name: bool,
    has_classification: bool,
) -> str:
    """
    Calculate a simple data quality score for a fund.

    Parameters
    ----------
    has_objective : bool
        Whether fund has objective text
    has_policy : bool
        Whether fund has investment policy
    has_trade_name : bool
        Whether fund has a trade name
    has_classification : bool
        Whether fund has ANBIMA classification

    Returns
    -------
    str
        Quality score: "High", "Medium", or "Low"
    """
    score = sum([has_objective, has_policy, has_trade_name, has_classification])

    if score >= 3:
        return "High"
    elif score >= 2:
        return "Medium"
    else:
        return "Low"


def truncate_text(text: Optional[str], max_length: int = 500) -> str:
    """
    Truncate text to a maximum length, adding ellipsis if needed.

    Parameters
    ----------
    text : str, optional
        Text to truncate
    max_length : int
        Maximum length (default 500)

    Returns
    -------
    str
        Truncated text
    """
    if not text:
        return ""

    if len(text) <= max_length:
        return text

    return text[:max_length - 3] + "..."


def validate_pinecone_index_name(name: str) -> bool:
    """
    Validate that an index name follows Pinecone naming rules.

    Rules:
    - Lowercase letters, numbers, hyphens only
    - Must start with a letter
    - Max 45 characters

    Parameters
    ----------
    name : str
        Index name to validate

    Returns
    -------
    bool
        True if valid, False otherwise
    """
    if not name:
        return False

    if len(name) > 45:
        logger.warning(f"Index name too long: {len(name)} chars (max 45)")
        return False

    if not name[0].isalpha():
        logger.warning(f"Index name must start with a letter: {name}")
        return False

    if not all(c.islower() or c.isdigit() or c == "-" for c in name):
        logger.warning(f"Index name contains invalid characters: {name}")
        return False

    return True


def batch_items(items: List, batch_size: int = 100) -> List[List]:
    """
    Split a list into batches of a specified size.

    Parameters
    ----------
    items : List
        List to batch
    batch_size : int
        Size of each batch (default 100)

    Returns
    -------
    List[List]
        List of batches
    """
    batches = []
    for i in range(0, len(items), batch_size):
        batches.append(items[i:i + batch_size])
    return batches


def format_score_percentage(score: float) -> str:
    """
    Format a similarity score as a percentage string.

    Parameters
    ----------
    score : float
        Score between 0 and 1

    Returns
    -------
    str
        Formatted percentage (e.g., "87.5%")
    """
    return f"{score * 100:.1f}%"


def create_filter_dict(
    investment_class: Optional[str] = None,
    fund_type: Optional[str] = None,
    structure: Optional[str] = None,
    target_audience: Optional[str] = None,
) -> Optional[Dict[str, str]]:
    """
    Create a metadata filter dictionary for Pinecone queries.

    Parameters
    ----------
    investment_class : str, optional
        Investment class filter
    fund_type : str, optional
        Fund type filter
    structure : str, optional
        Structure filter
    target_audience : str, optional
        Target audience filter

    Returns
    -------
    dict or None
        Filter dictionary or None if no filters
    """
    filters = {}

    if investment_class:
        filters["investment_class"] = investment_class
    if fund_type:
        filters["fund_type"] = fund_type
    if structure:
        filters["structure"] = structure
    if target_audience:
        filters["target_audience"] = target_audience

    return filters if filters else None
