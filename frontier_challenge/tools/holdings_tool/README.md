# Holdings Search Tool

Find funds that hold specific companies or assets in their portfolios.

## Purpose

Enables discovery of funds based on their portfolio holdings using natural language queries. Supports fuzzy matching to handle variations in company and asset names.

## Components

### holdings_tool.py
Main tool implementation with:
- Natural language entity extraction
- Fuzzy matching using Levenshtein distance
- Portfolio weight filtering
- Grouped or detailed result views

### models.py
Pydantic models:
- `HoldingsSearchCriteria` - Structured search parameters
- `HoldingRecord` - Individual holding details
- `FundSummaryWithHolding` - Fund-level aggregated view
- `EntityExtractionResult` - Extracted company/asset names

### holdings_prompt.py
LLM prompts for entity extraction from natural language queries.

## Design Decisions

**Two-Stage Process**:
1. Extract entity names from natural language using LLM
2. Perform fuzzy matching in DuckDB using Levenshtein distance

This separates natural language understanding from data matching.

**Fuzzy Matching**: Uses DuckDB's native Levenshtein distance function to handle:
- Typos and misspellings
- Abbreviations (e.g., "VALE" vs "Vale S.A.")
- Different formatting (e.g., "Petrobras" vs "PETROBRAS S.A.")

**Configurable Threshold**: Default Levenshtein distance threshold of 3 balances precision and recall. Adjustable based on use case.

**Multiple View Options**: Supports both grouped (by fund) and detailed (all holdings) views to accommodate different query needs.

**Portfolio Weight Filtering**: Optional minimum weight threshold to focus on significant holdings rather than token positions.

## Usage

```python
from frontier_challenge.tools import HoldingsSearchTool

tool = HoldingsSearchTool(
    db_path="data/br_funds.db",
    model_name="gpt-4o-mini"
)

# Natural language query
result = tool.search_by_natural_language(
    "Funds that invest in Petrobras",
    group_by_fund=True,
    min_weight_pct=1.0
)
```

## Search Process

1. User query → "Funds investing in Apple"
2. LLM extracts entity → "Apple Inc."
3. DuckDB fuzzy matches → "APPLE INC", "Apple Computer", etc.
4. Returns funds with matching holdings
5. Optionally groups by fund with aggregated statistics
