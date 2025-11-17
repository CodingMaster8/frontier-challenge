# Frontier Challenge

Main package for the Brazilian Investment Funds AI Agent system.

## Overview

This package implements a production-grade conversational AI agent that enables natural language interaction with a comprehensive database of Brazilian investment funds. The system uses LangGraph for agent orchestration and provides multiple specialized tools for fund discovery and analysis.

## Architecture

The system follows a modular architecture with clear separation of concerns:

- `agent/` - Core agent implementation using LangGraph
- `db/` - Database view management and setup
- `ingest/` - ETL pipeline for CVM data ingestion
- `tools/` - Specialized tools for fund search and analysis
- `settings.py` - Environment configuration and API keys

## Key Features

- Natural language query understanding
- Multi-tool orchestration with intelligent routing
- Bilingual support (Portuguese/English)
- Type-safe data models using Pydantic
- Comprehensive error handling and retry logic
- Conversational memory management

## Design Decisions

**LangGraph State Machine**: Uses LangGraph instead of sequential chains to enable complex conversation flows with branching logic, tool selection, and error recovery.

**View-Based Architecture**: Leverages DuckDB views to optimize query performance and maintain clean separation between raw data and application logic.

**Specialized Tools**: Each tool focuses on a specific capability (semantic search, structured filters, holdings analysis, visualizations, CNPJ lookup) to maintain modularity and testability.

**Type Safety**: Extensive use of Pydantic models ensures data validation and type safety across the entire pipeline.

## Usage

```python
from frontier_challenge.agent import get_financial_agent_graph

# Initialize agent
graph = get_financial_agent_graph(
    db_path="data/br_funds.db",
    model_name="gpt-4o-mini"
)

# Run query
response = graph.invoke({
    "messages": [{"role": "user", "content": "Find sustainable funds"}]
})
```

## Dependencies

- DuckDB for analytical database
- LangChain/LangGraph for agent orchestration
- OpenAI for LLM and embeddings
- Pinecone for vector search
- Pandas for data processing
