"""
Pydantic models for Semantic Search Tool

Provides type-safe data structures for semantic search queries and results.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


# ============================================================================
# Search Query Models
# ============================================================================


class SemanticSearchQuery(BaseModel):
    """Input query for semantic search"""

    query: str = Field(..., description="Natural language search query")
    top_k: int = Field(10, description="Number of results to return", ge=1, le=100)
    filter_by: Optional[dict] = Field(
        None,
        description="Metadata filters, e.g., {'investment_class': 'Ações'}"
    )
    use_rerank: bool = Field(False, description="Enable Cohere reranking")

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        """Validate query is not empty"""
        if not v or not v.strip():
            raise ValueError("Query cannot be empty")
        return v.strip()

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, v: int) -> int:
        """Validate top_k is reasonable"""
        if v < 1:
            raise ValueError("top_k must be at least 1")
        if v > 100:
            raise ValueError("top_k cannot exceed 100")
        return v


# ============================================================================
# Result Models
# ============================================================================


class SemanticSearchMatch(BaseModel):
    """A single fund match from semantic search"""

    cnpj: str = Field(..., description="Fund CNPJ (unique identifier)")
    score: float = Field(..., description="Similarity score (0-1)", ge=0, le=1)
    legal_name: str = Field(..., description="Fund legal name")
    trade_name: Optional[str] = Field(None, description="Fund trade name")
    investment_class: Optional[str] = Field(None, description="Investment class")
    anbima_classification: Optional[str] = Field(None, description="ANBIMA classification")
    fund_type: Optional[str] = Field(None, description="Fund type")
    structure: Optional[str] = Field(None, description="Fund structure")
    target_audience: Optional[str] = Field(None, description="Target audience")
    data_quality: Optional[str] = Field(None, description="Data quality score")
    matched_text_preview: Optional[str] = Field(None, description="Preview of matched text")

    # Optional reranking score
    rerank_score: Optional[float] = Field(None, description="Cohere rerank score")
    original_score: Optional[float] = Field(None, description="Original similarity score before reranking")

    class Config:
        frozen = False  # Allow modifications for reranking


class SemanticSearchResult(BaseModel):
    """Result from a semantic search operation"""

    success: bool = Field(True, description="Whether search was successful")
    matches: List[SemanticSearchMatch] = Field(default_factory=list, description="List of matching funds")
    total_matches: int = Field(0, description="Total number of matches returned")
    query: str = Field(..., description="Original search query")
    embedding_model: Optional[str] = Field(None, description="Embedding model used")
    vector_dimension: Optional[int] = Field(None, description="Embedding dimension")
    reranked: bool = Field(False, description="Whether results were reranked")
    execution_time_ms: Optional[float] = Field(None, description="Execution time in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if search failed")

    class Config:
        arbitrary_types_allowed = True


# ============================================================================
# Index Stats Models
# ============================================================================


class IndexStats(BaseModel):
    """Statistics about the Pinecone index"""

    total_vectors: int = Field(..., description="Total number of vectors in index")
    dimension: int = Field(..., description="Embedding dimension")
    index_fullness: float = Field(..., description="Index fullness (0-1)", ge=0, le=1)
    metric: str = Field(..., description="Distance metric (cosine, euclidean, etc.)")
    index_name: str = Field(..., description="Pinecone index name")


class BuildIndexResult(BaseModel):
    """Result from building the index"""

    success: bool = Field(True, description="Whether index build was successful")
    total_vectors: int = Field(0, description="Number of vectors indexed")
    index_name: str = Field(..., description="Index name")
    rebuild: bool = Field(False, description="Whether index was rebuilt from scratch")
    execution_time_ms: Optional[float] = Field(None, description="Build time in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if build failed")
