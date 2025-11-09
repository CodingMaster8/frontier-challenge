# SQL Views for Hybrid Fund Search System

## Overview

This directory contains SQL views that power a **hybrid search architecture** for Brazilian investment fund discovery. The system combines **semantic search (vector embeddings)** and **structured filtering (text-to-SQL)** to handle diverse query types.

## Architecture: Why 3 Views?

Instead of denormalizing all data into a single table, we created **3 specialized views** optimized for different search scenarios:

```
User Query → AI Router → Select Tool(s) → Query Appropriate View(s) → Return CNPJs
```

### The Problem with a Single Table
- **Different temporal dimensions**: Funds have monthly snapshots, performance has monthly time-series, positions change over time
- **Different cardinalities**: One fund → many positions → many assets
- **Update anomalies**: Denormalization would create massive duplication
- **Query inefficiency**: Semantic search doesn't need numeric data, SQL filters don't need text descriptions

### The Solution: Specialized Views
Each view is optimized for specific query patterns and tool implementations.

---

## View 1: `fund_semantic_search_view`

### Purpose
Optimized for **fuzzy/conceptual queries** using natural language and embedding similarity.

### When to Use
- "What is the Bradesco gold fund?"
- "Show me sustainable investing funds"
- "Funds that invest in Latin American tech"
- "ESG-focused equity funds"

### Key Features
- Combines all searchable text into a single `searchable_text` field
- Includes Portuguese descriptions (OBJETIVO, POLIT_INVEST, NM_FANTASIA)
- Includes English metadata (investment_class, anbima_classification)
- Only latest snapshot per fund (performance optimization)
- Only ACTIVE funds

### Important Columns
- `cnpj`: Brazilian tax ID (this is what you return!)
- `legal_name`: Official fund name
- `trade_name`: Commercial/marketing name
- `searchable_text`: Concatenated text for embedding generation
- `objective`: Portuguese investment objective
- `investment_policy`: Portuguese policy description

### Usage Pattern
```python
# 1. Generate embeddings for searchable_text
# 2. Store embeddings in a separate table or column
# 3. On query: Generate embedding for user query
# 4. Find top-K most similar funds using cosine similarity
# 5. Return CNPJs
```

### Sample Query
```sql
SELECT cnpj, legal_name, investment_class, searchable_text
FROM fund_semantic_search_view
WHERE LOWER(searchable_text) LIKE '%bradesco%'
LIMIT 10;
```

---

## View 2: `fund_structured_filter_view`

### Purpose
Optimized for **precise structured queries** with numeric filters, comparisons, and rankings.

### When to Use
- "Funds with >15% YTD returns"
- "Equity funds with minimum investment under R$1,000"
- "Multimercado funds that beat CDI by >5%"
- "Low-volatility fixed income funds"
- "Funds ranked by Sharpe ratio"

### Key Features
- All numeric fields properly cast (fees, returns, NAV, minimums)
- Calculated performance metrics (YTD, 12M, 6M, 3M, last month)
- Risk metrics (volatility, Sharpe ratio, negative months)
- Excess returns vs benchmark
- Investment constraints (minimums, lockup, liquidity)

### Important Columns
**Classification:**
- `investment_class`: Ações, Renda Fixa, Multimercado, Cambial
- `anbima_classification`: More granular ANBIMA categories
- `structure`: OPEN (open-end) vs CLOSED (closed-end)
- `target_audience`: Retail, Qualified, Professional

**Performance:**
- `return_ytd_2024_avg`: Average monthly return YTD 2024
- `return_12m_avg`: 12-month average return
- `return_5y_pct`: 5-year cumulative return
- `excess_return_5y_pct`: Return vs benchmark
- `volatility_12m`: Standard deviation of returns

**Financials:**
- `nav`: Net Asset Value (fund size)
- `management_fee_pct`: Annual management fee
- `performance_fee_pct`: Performance fee
- `min_initial_investment`: Minimum to open account

**Liquidity:**
- `lockup_days`: Lock-up period
- `redemption_payment_days`: Days to receive money after redemption

### Usage Pattern
```python
# Text-to-SQL Tool:
# 1. Parse user query into SQL WHERE/ORDER BY clauses
# 2. Execute filtered query on this view
# 3. Return matching CNPJs with relevant metrics
```

### Sample Queries
```sql
-- High-return equity funds
SELECT cnpj, legal_name, return_ytd_2024_avg, management_fee_pct
FROM fund_structured_filter_view
WHERE investment_class = 'Ações'
  AND return_ytd_2024_avg > 15.0
ORDER BY return_ytd_2024_avg DESC
LIMIT 10;

-- Accessible funds (low minimum)
SELECT cnpj, legal_name, min_initial_investment, return_12m_avg
FROM fund_structured_filter_view
WHERE min_initial_investment <= 1000
  AND min_initial_investment IS NOT NULL
ORDER BY return_12m_avg DESC;

-- Funds beating their benchmark
SELECT cnpj, legal_name, return_5y_pct, benchmark_return_5y_pct, excess_return_5y_pct
FROM fund_structured_filter_view
WHERE excess_return_5y_pct > 5.0
ORDER BY excess_return_5y_pct DESC;
```

---

## View 3: `fund_portfolio_analysis_view`

### Purpose
Optimized for **portfolio-based queries** - searching by what funds actually hold.

### When to Use
- "Funds that invest in tech stocks"
- "Funds with USD currency exposure"
- "Funds holding Petrobras"
- "Funds investing in government bonds"
- "Well-diversified international funds"

### Key Features
- Aggregates portfolio composition by asset class, instrument, country, currency
- Provides top holdings preview
- Shows diversification metrics (number of positions, countries, asset classes)
- Links to detailed holdings via companion view

### Important Columns
- `total_positions`: Number of holdings in portfolio
- `num_asset_classes`: Diversification across asset types
- `num_countries`: Geographic diversification
- `top_5_holdings`: Preview of largest positions
- `asset_class_breakdown`: "EQUITY:40% | FIXED_INCOME:35% | CASH:25%"
- `country_exposure`: "BRA:60% | USA:25% | EUR:15%"
- `currency_exposure`: "BRL:70% | USD:20% | EUR:10%"
- `instrument_breakdown`: Detailed instrument types

### Companion View: `fund_holdings_detail_view`
For granular asset-level searches:
- Individual positions with weights
- Specific asset names and issuers
- Filter by specific companies, sectors, or instruments

### Usage Pattern
```python
# Portfolio Analysis Tool:
# 1. Parse user query for asset types, countries, instruments
# 2. Query portfolio view with LIKE/JOIN conditions
# 3. Can combine with semantic search on asset descriptions
# 4. Return funds with matching portfolio composition
```

### Sample Queries
```sql
-- Funds with equity exposure
SELECT cnpj, legal_name, asset_class_breakdown, total_positions
FROM fund_portfolio_analysis_view
WHERE asset_class_breakdown LIKE '%EQUITY%'
ORDER BY total_positions DESC
LIMIT 10;

-- Funds with USD exposure
SELECT cnpj, legal_name, currency_exposure
FROM fund_portfolio_analysis_view
WHERE currency_exposure LIKE '%USD%'
ORDER BY legal_name;

-- Find funds holding Petrobras (detailed view)
SELECT cnpj, legal_name, asset_name, portfolio_weight_pct
FROM fund_holdings_detail_view
WHERE LOWER(asset_name) LIKE '%petrobras%'
   OR LOWER(issuer_name) LIKE '%petrobras%'
ORDER BY portfolio_weight_pct DESC
LIMIT 10;

-- International diversification
SELECT cnpj, legal_name, num_countries, country_exposure
FROM fund_portfolio_analysis_view
WHERE num_countries > 3
ORDER BY num_countries DESC;
```

---

## Setup & Testing

### 1. Create the Views
```bash
# Using the test script (recommended)
uv run python scripts/test_views.py

# Or manually with DuckDB CLI (if you have it installed)
duckdb data/br_funds.db < sql/00_create_all_views.sql
```

### 2. Verify Views Were Created
```python
import duckdb
conn = duckdb.connect('data/br_funds.db')
print(conn.execute("SHOW TABLES").fetchall())
```

### 3. Test Sample Queries
Run the examples provided in each SQL file's comment section.

---

## Design Decisions ✅

### What We DID:
- ✅ **Latest snapshot only** - Filter by `MAX(timestamp)` to avoid historical duplicates
- ✅ **Only ACTIVE funds** - Filter `status = 'ACTIVE'` to exclude cancelled funds
- ✅ **CNPJ extraction** - Consistently extract from `identifiers[1].value`
- ✅ **Both names** - Include both `legal_name` (official) and `NM_FANTASIA` (trade name)
- ✅ **Portuguese text** - Rich semantic search with OBJETIVO and POLIT_INVEST
- ✅ **Type casting** - All numeric fields cast with `TRY_CAST` to handle nulls
- ✅ **Separate views** - Optimized for different query patterns

### What We DIDN'T DO:
- ❌ **No denormalization** - Keep time-series data separate
- ❌ **No all snapshots** - Would make searches exponentially slower
- ❌ **No embedding in view** - Embeddings generated separately and stored
- ❌ **No complex joins in semantic view** - Keep text concatenation simple

---

## Next Steps

### Phase 1: Semantic Search Tool
1. Generate embeddings for `searchable_text` using sentence-transformers
2. Store embeddings (DuckDB doesn't have native vector support, so use external vector DB or store as arrays)
3. Implement cosine similarity search
4. Build tool wrapper that takes query → returns CNPJs

### Phase 2: Structured Filter Tool (Text-to-SQL)
1. Build LLM-based text-to-SQL converter
2. Use view schema as context for SQL generation
3. Add safety guards (read-only, query validation)
4. Build tool wrapper with structured parameters

### Phase 3: Portfolio Analysis Tool
1. Similar to structured filter but focused on holdings
2. Can use semantic search on `financial_instrument_description`
3. Combine with asset name matching

### Phase 4: Router + Agent
1. LLM function calling to choose tool(s)
2. Route queries to appropriate view(s)
3. Combine results if multiple tools used
4. Format output with CNPJs + relevant metadata

---

## Performance Notes

- **View refresh**: These are regular views (not materialized), so they compute on each query
- **For production**: Consider materializing views and refreshing daily/weekly
- **Indexing**: Add indexes on `cnpj`, `fund_id`, `timestamp` for faster joins
- **Embeddings**: Store separately for better performance (DuckDB arrays are not optimized for similarity search)

---

## Files in This Directory

```
sql/
├── 00_create_all_views.sql              # Master script to create all views
├── 01_create_semantic_search_view.sql   # View 1: Semantic search
├── 02_create_structured_filter_view.sql # View 2: Structured filtering
├── 03_create_portfolio_analysis_view.sql # View 3: Portfolio analysis
└── README.md                             # This file
```

---

## Questions?

- **Why not use DuckDB's vector extension?** At the time of writing, it's experimental. Easier to use external vector store (Chroma, Pinecone) or simple cosine similarity.
- **Why separate holdings detail view?** Performance - the main view aggregates, detail view is for deep dives.
- **Can I combine queries across views?** Yes! The router can call multiple tools and merge results.
- **What about caching?** Views compute on-demand. Add materialized views or application-layer caching for production.

---

**Author**: GitHub Copilot
**Date**: November 2024
**Purpose**: Frontier AI Challenge - Brazilian Fund Search System
