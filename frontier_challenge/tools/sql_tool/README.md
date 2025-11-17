# Structured Filter Tool (SQL Tool)

Text-to-SQL conversion for structured fund filtering queries.

## Purpose

Converts natural language queries with specific numeric criteria into SQL queries against the structured filter view. Handles complex conditions, comparisons, and aggregations.

## Components

### structured_filter_tool.py
Main tool class with synchronous interface to the LangGraph workflow.

### graph.py
LangGraph state machine implementing:
- Schema extraction and context building
- Text-to-SQL conversion with validation
- Query execution and error recovery
- Automatic retry logic

### models.py
Pydantic models:
- `FundFilterCriteria` - Structured filter parameters
- `FundRecord` - Individual fund record
- `FilterResult` - Query results with metadata
- `FilterQueryState` - Graph state management

### structured_filter_prompts.py
System prompts for SQL generation with schema context and best practices.

### utils.py
Helper functions for SQL validation, error parsing, and query formatting.

## Design Decisions

**LangGraph for Orchestration**: Uses state machine rather than simple LLM call to enable:
- Query validation before execution
- Automatic retry on SQL errors
- Error feedback loop to LLM
- Structured error handling

**Schema-Aware Generation**: Provides full schema context to LLM including:
- Column names and types
- Sample values
- Valid ranges
- Common patterns

This reduces hallucination and improves SQL quality.

**View-Based Queries**: All queries target `fund_structured_filter_view` which provides:
- Pre-computed performance metrics
- Clean column names
- Consistent data types
- Optimized joins

**SQL Validation**: Validates generated SQL before execution to catch:
- Syntax errors
- Invalid column names
- Type mismatches
- Dangerous operations (no DELETE, DROP, etc.)

**Result Limiting**: Automatically limits results to prevent overwhelming responses while allowing configurable limits via natural language.

**Error Recovery**: When SQL fails:
1. Captures error message
2. Feeds back to LLM with error context
3. LLM generates corrected query
4. Retries up to maximum attempts

## Usage

```python
from frontier_challenge.tools import StructuredFilterTool

tool = StructuredFilterTool(
    db_path="data/br_funds.db",
    model_name="gpt-4o-mini"
)

result = tool.filter_funds(
    "Funds with >15% YTD return and <2% fees"
)
```

## Example Queries

- "Top 10 funds by 12-month return"
- "Equity funds with AUM > R$100 million"
- "Funds with Sharpe ratio above 1.5"
- "Low volatility funds that beat benchmark"
- "Funds with fees below 1% and positive 6M return"

## SQL Generation Process

1. User query → Natural language
2. Extract schema context → Table structure
3. LLM generates SQL → SELECT statement
4. Validate SQL → Syntax and safety checks
5. Execute query → DuckDB
6. Return results → Structured Pydantic models

If errors occur at step 5, feedback to step 3 with error details.
