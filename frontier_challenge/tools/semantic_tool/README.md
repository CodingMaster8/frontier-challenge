# Semantic Search Tool

Vector-based semantic search for natural language fund discovery.

## Purpose

Enables fuzzy, conceptual fund discovery using natural language queries. Converts text to vector embeddings and finds semantically similar funds based on descriptions, strategies, and metadata.

## Components

### semantic_search_tool.py
Main tool implementation with:
- Query embedding generation
- Pinecone vector search
- Cohere reranking for improved relevance
- Index building and management
- Type-safe results

### models.py
Pydantic models:
- `SemanticSearchQuery` - Query parameters
- `SemanticSearchMatch` - Individual fund match with score
- `SemanticSearchResult` - Complete search result
- `BuildIndexResult` - Index building results

### utils.py
Helper functions for Pinecone index name validation and metadata handling.

## Design Decisions

**Pinecone for Vector Storage**: Uses managed Pinecone service instead of local vector database because:
- Production-grade infrastructure with high availability
- Efficient similarity search at scale
- Free tier sufficient for ~100K funds
- No local infrastructure to maintain

**OpenAI Embeddings**: Uses `text-embedding-3-small` (1536 dimensions) for:
- Strong multilingual support (Portuguese/English)
- Cost-effective compared to larger models
- Good balance of quality and performance

**Cohere Reranking**: Optional reranking stage improves relevance by:
- Understanding query intent beyond vector similarity
- Handling nuanced queries better
- Minimal latency impact with significant quality gain

**Rich Metadata Storage**: Stores comprehensive fund metadata in Pinecone to avoid database lookups:
- Fund names, types, managers
- Performance metrics
- Fees and AUM
- Enables filtering and sorting without additional queries

**Batch Index Building**: Processes embeddings in batches for efficient index creation from large datasets.

## Usage

```python
from frontier_challenge.tools import SemanticSearchTool

tool = SemanticSearchTool(
    db_path="data/br_funds.db",
    index_name="br-funds-prod"
)

# Search
result = tool.search(
    query="sustainable technology funds",
    top_k=10,
    use_rerank=True
)

# Build/rebuild index
result = tool.build_index(batch_size=100)
```

## Search Process

1. User query → "sustainable ESG funds"
2. OpenAI embeds query → 1536-dim vector
3. Pinecone finds similar vectors → top 50 candidates
4. Cohere reranks → top 10 most relevant
5. Returns sorted results with similarity scores

## Index Management

The tool provides methods for:
- Building new indexes from database
- Checking index statistics
- Validating index health
- Deleting indexes
