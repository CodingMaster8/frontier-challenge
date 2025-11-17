# CNPJ Lookup Tool

Fast retrieval of fund information using CNPJ identifiers.

## Purpose

Provides instant lookup of fund details by CNPJ number, supporting both single and batch queries. This tool is optimized for direct retrieval when the exact fund identifier is known.

## Components

### cnpj_tool.py
Main tool implementation with:
- CNPJ validation and normalization
- Single and batch lookup support
- Direct SQL queries for performance
- Type-safe results

### models.py
Pydantic models:
- `FundInfo` - Core fund attributes
- `CNPJLookupResult` - Complete lookup result with metadata

## Design Decisions

**Direct Queries**: Uses simple SQL SELECT statements instead of complex search algorithms for maximum performance when CNPJ is known.

**CNPJ Normalization**: Accepts CNPJs with or without formatting (dots, slashes, dashes) and normalizes them for consistent matching.

**Batch Support**: Efficiently handles multiple CNPJs in a single query using SQL IN clause rather than multiple round trips.

**View-Based**: Queries the `fund_semantic_search_view` which provides pre-joined fund data with all relevant fields.

## Usage

```python
from frontier_challenge.tools import CNPJLookupTool

tool = CNPJLookupTool(db_path="data/br_funds.db")

# Single lookup
result = tool.lookup_cnpj("12.345.678/0001-90")

# Batch lookup
result = tool.lookup_cnpj(["12.345.678/0001-90", "98.765.432/0001-10"])
```

## Output Fields

Each fund result includes:
- CNPJ and legal name
- Fund type and classification
- Manager and administrator
- Performance metrics (returns, volatility, Sharpe ratio)
- Fees and AUM
- Benchmark information
