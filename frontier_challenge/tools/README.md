# Tools Module

Specialized tools for fund search, analysis, and visualization.

## Purpose

This module provides a suite of specialized tools that enable different types of queries and analyses on the Brazilian fund database. Each tool is designed to handle a specific query pattern efficiently.

## Tool Overview

### CNPJ Lookup Tool (`cnpj_tool/`)
Fast lookup of fund information by CNPJ identifier(s).

**Use Cases**:
- Direct fund lookup by CNPJ
- Batch retrieval of multiple funds
- Quick reference for specific funds

**Design**: Simple SQL queries against the semantic search view for instant results.

### Holdings Search Tool (`holdings_tool/`)
Find funds that invest in specific companies or assets.

**Use Cases**:
- "Funds that invest in Petrobras"
- "Funds holding Apple stock"
- "Funds with exposure to Vale"

**Design**: Combines entity extraction using LLM with fuzzy matching (Levenshtein distance) in DuckDB to handle variations in company names.

### Semantic Search Tool (`semantic_tool/`)
Natural language search using vector embeddings.

**Use Cases**:
- "Bradesco gold fund"
- "sustainable technology investing"
- "fundos de renda fixa conservadores"

**Design**: Uses OpenAI embeddings (text-embedding-3-small) with Pinecone vector database and Cohere reranking for production-grade semantic search.

### Structured Filter Tool (`sql_tool/`)
Text-to-SQL conversion for structured queries.

**Use Cases**:
- "Funds with >15% YTD return and <2% fees"
- "Large cap equity funds with >R$100M AUM"
- "Funds with low volatility and positive Sharpe ratio"

**Design**: LangGraph-based text-to-SQL pipeline with query validation, error recovery, and automatic retry logic.

### Visualization Tool (`viz_tool/`)
Automatic generation of financial charts and graphs.

**Use Cases**:
- Performance comparisons
- Fee distributions
- Risk-return scatter plots

**Design**: Multi-stage LangGraph workflow that analyzes data, proposes visualizations, generates Python code, and executes with error recovery.

## Design Decisions

**Tool Specialization**: Each tool focuses on a specific query pattern rather than creating one generic tool. This improves:
- Performance (each tool is optimized for its use case)
- Reliability (simpler code paths)
- Maintainability (clear boundaries)
- Testability (isolated components)

**Type Safety**: All tools use Pydantic models for inputs and outputs, ensuring validation and type safety across the system.

**Error Handling**: Each tool implements comprehensive error handling with informative error messages and recovery strategies.

**Database Views**: Tools query pre-built views rather than raw tables to ensure consistent data models and optimized performance.

## Common Patterns

All tools follow similar patterns:
- Constructor accepts database path and configuration
- Main execution method takes natural language input
- Returns strongly-typed Pydantic model results
- Includes logging for debugging and monitoring
- Implements error handling and validation
