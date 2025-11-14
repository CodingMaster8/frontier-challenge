"""
Semantic Search Tool for Brazilian Investment Funds - Pinecone Edition

This tool uses vector embeddings to enable natural language search across funds.
It supports Portuguese and English queries for fuzzy/conceptual fund discovery.

Uses:
- OpenAI text-embedding-3-small for embeddings (1536 dimensions)
- Pinecone for vector storage (free tier: 2GB, ~100K vectors)
- ReRank Cohere algorithm for improved search relevance

Example queries:
- "Bradesco gold fund"
- "sustainable technology investing"
- "fundos de renda fixa conservadores"
- "Latin American equity exposure"
"""

import logging
import os
from typing import List, Dict, Optional
import time
from datetime import datetime

from dotenv import load_dotenv

import duckdb
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from .models import (
    SemanticSearchQuery,
    SemanticSearchMatch,
    SemanticSearchResult,
    IndexStats,
    BuildIndexResult,
)
from .utils import validate_pinecone_index_name

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class SemanticSearchTool:
    """
    Production-grade semantic search tool using OpenAI embeddings and Pinecone.

    This tool converts natural language queries to vector embeddings and searches
    for semantically similar funds using cosine similarity.

    Requirements:
    - OPENAI_API_KEY environment variable
    - PINECONE_API_KEY environment variable
    - COHERE_API_KEY environment variable (optional, for reranking)
    """

    def __init__(
        self,
        db_path: str = "data/br_funds.db",
        index_name: str = "br-funds-search",
        embedding_model: str = "text-embedding-3-large",
        dimension: int = 3072,  # 3072 for large, 1536 for small
        metric: str = "cosine",
        use_rerank: bool = False,
    ):
        """
        Initialize semantic search tool.

        Parameters
        ----------
        db_path : str
            Path to DuckDB database with fund views
        index_name : str
            Pinecone index name (lowercase, no spaces)
        embedding_model : str
            OpenAI embedding model. Options:
            - "text-embedding-3-small" (1536D, cheap, fast)
            - "text-embedding-3-large" (3072D, better quality)
        dimension : int
            Embedding dimension (1536 for small, 3072 for large)
        metric : str
            Distance metric: "cosine", "euclidean", or "dotproduct"
        use_rerank : bool
            Enable Cohere reranking (default False)
        """
        self.db_path = db_path
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.metric = metric
        self.use_rerank = use_rerank

        # Validate index name
        if not validate_pinecone_index_name(index_name):
            raise ValueError(
                f"Invalid Pinecone index name: {index_name}. "
                "Must be lowercase letters, numbers, hyphens; start with letter; max 45 chars."
            )

        # Initialize OpenAI client
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")

        logger.info(f"Initializing OpenAI client with model: {embedding_model}")
        self.openai_client = OpenAI(api_key=api_key)

        # Initialize Pinecone
        pinecone_key = os.getenv("PINECONE_API_KEY")
        if not pinecone_key:
            raise ValueError("PINECONE_API_KEY environment variable not set")

        logger.info("Initializing Pinecone client")
        self.pc = Pinecone(api_key=pinecone_key)
        self.index = None

        # Initialize Cohere for reranking
        if use_rerank:
            try:
                import cohere
                cohere_key = os.getenv("COHERE_API_KEY")
                if cohere_key:
                    self.cohere_client = cohere.Client(api_key=cohere_key)
                    logger.info("Cohere reranking enabled")
                else:
                    logger.warning("COHERE_API_KEY not set, reranking disabled")
                    self.use_rerank = False
                    self.cohere_client = None
            except ImportError:
                logger.warning("Cohere package not installed, reranking disabled")
                self.use_rerank = False
                self.cohere_client = None
        else:
            self.cohere_client = None

        logger.info(f"Initialized SemanticSearchTool with db: {db_path}")

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using OpenAI."""
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding

    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batch processing)."""

        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            logger.info(f"Generating embeddings for batch {i//batch_size + 1} ({len(batch)} texts)...")

            response = self.openai_client.embeddings.create(
                input=batch,
                model=self.embedding_model
            )

            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            # Rate limiting: sleep briefly between batches
            if i + batch_size < len(texts):
                time.sleep(0.5)

        return all_embeddings

    def build_index(self, force_rebuild: bool = False) -> BuildIndexResult:
        """
        Build or load the Pinecone vector index from fund data.

        Parameters
        ----------
        force_rebuild : bool
            If True, delete and rebuild index. Default False.

        Returns
        -------
        BuildIndexResult
            Result with index statistics and build status
        """
        start_time = datetime.now()

        try:
            # Check if index exists
            existing_indexes = [idx.name for idx in self.pc.list_indexes()]

            if self.index_name in existing_indexes:
                if not force_rebuild:
                    logger.info(f"Loading existing index: {self.index_name}")
                    self.index = self.pc.Index(self.index_name)
                    stats = self.index.describe_index_stats()
                    count = stats.total_vector_count
                    logger.info(f"Loaded index with {count:,} vectors")

                    execution_time = (datetime.now() - start_time).total_seconds() * 1000

                    return BuildIndexResult(
                        success=True,
                        total_vectors=count,
                        index_name=self.index_name,
                        rebuild=False,
                        execution_time_ms=execution_time
                    )
                else:
                    logger.info(f"Deleting existing index for rebuild: {self.index_name}")
                    self.pc.delete_index(self.index_name)
                    # Wait for deletion to complete
                    time.sleep(5)

            # Create new index
            logger.info(f"Creating new Pinecone index: {self.index_name}")
            logger.info(f"Dimension: {self.dimension}, Metric: {self.metric}")

            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'  # Free tier region
                )
            )

            # Wait for index to be ready
            logger.info("Waiting for index to be ready...")
            while not self.pc.describe_index(self.index_name).status['ready']:
                time.sleep(1)

            self.index = self.pc.Index(self.index_name)
            logger.info("Index created and ready")

            # Load funds from database
            logger.info("Loading funds from database view...")
            conn = duckdb.connect(self.db_path, read_only=True)

            funds_df = conn.execute("""
                SELECT
                    cnpj,
                    legal_name,
                    trade_name,
                    investment_class,
                    anbima_classification,
                    fund_type,
                    structure,
                    target_audience,
                    objective,
                    investment_policy,
                    searchable_text,
                    data_quality
                FROM fund_semantic_search_view
                ORDER BY data_quality DESC, cnpj;
            """).fetchdf()

            conn.close()

            logger.info(f"Loaded {len(funds_df):,} funds from view")

            # Prepare texts for embedding
            texts = []
            for idx, row in funds_df.iterrows():
                # Use searchable_text for embedding (optimized concatenation)
                text = row['searchable_text'] or row['legal_name']
                texts.append(text)

            # Generate embeddings in batches
            logger.info("Generating embeddings with OpenAI...")
            embeddings = self._generate_embeddings_batch(texts)
            logger.info(f"Generated {len(embeddings):,} embeddings")

            # Prepare vectors for Pinecone
            vectors = []
            for idx, row in funds_df.iterrows():
                metadata = {
                    'cnpj': row['cnpj'],
                    'legal_name': row['legal_name'],
                    'trade_name': row['trade_name'] or '',
                    'investment_class': row['investment_class'] or '',
                    'anbima_classification': row['anbima_classification'] or '',
                    'fund_type': row['fund_type'] or '',
                    'structure': row['structure'] or '',
                    'target_audience': row['target_audience'] or '',
                    'data_quality': row['data_quality'],
                    'searchable_text_preview': (row['searchable_text'] or '')[:500],  # First 500 chars
                }

                vectors.append({
                    'id': row['cnpj'],
                    'values': embeddings[idx],
                    'metadata': metadata
                })

            # Upload to Pinecone in batches
            logger.info("Uploading vectors to Pinecone...")
            batch_size = 100
            for i in range(0, len(vectors), batch_size):
                batch = vectors[i:i + batch_size]
                self.index.upsert(vectors=batch)
                logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(vectors)-1)//batch_size + 1}")

            # Wait for upserts to be reflected
            time.sleep(2)

            # Get final count
            stats = self.index.describe_index_stats()
            total_count = stats.total_vector_count

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            logger.info(f"✅ Index built successfully! {total_count:,} funds indexed")
            logger.info(f"📊 Index stats: {stats}")

            return BuildIndexResult(
                success=True,
                total_vectors=total_count,
                index_name=self.index_name,
                rebuild=force_rebuild,
                execution_time_ms=execution_time
            )

        except Exception as e:
            logger.error(f"Error building index: {e}")
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return BuildIndexResult(
                success=False,
                total_vectors=0,
                index_name=self.index_name,
                rebuild=force_rebuild,
                execution_time_ms=execution_time,
                error_message=str(e)
            )

    def _rerank_results(
        self,
        query: str,
        candidates: List[SemanticSearchMatch],
        top_k: int = 7
    ) -> List[SemanticSearchMatch]:
        """
        Rerank search results using Cohere.

        Parameters
        ----------
        query : str
            Original search query
        candidates : List[SemanticSearchMatch]
            Initial search results from Pinecone
        top_k : int
            Number of results to return after reranking

        Returns
        -------
        List[SemanticSearchMatch]
            Reranked results with rerank_score added
        """
        if not self.cohere_client or len(candidates) <= top_k:
            return candidates[:top_k]

        # Prepare documents for reranking
        documents = []
        for c in candidates:
            # Create rich text representation
            doc = f"{c.legal_name}"
            if c.trade_name:
                doc += f" ({c.trade_name})"
            doc += f" | {c.investment_class} | {c.anbima_classification}"
            if c.matched_text_preview:
                doc += f" | {c.matched_text_preview[:200]}"
            documents.append(doc)

        # Rerank with Cohere
        logger.debug(f"Reranking {len(candidates)} candidates with Cohere...")
        try:
            rerank_response = self.cohere_client.rerank(
                model="rerank-v3.5",
                query=query,
                documents=documents,
                top_n=top_k,
                return_documents=False
            )

            # Add rerank scores to results
            reranked = []
            for result in rerank_response.results:
                idx = result.index
                candidate = candidates[idx]
                # Update the match with rerank scores
                candidate.rerank_score = result.relevance_score
                candidate.original_score = candidate.score  # Keep original
                reranked.append(candidate)

            logger.debug(f"Reranked to top {len(reranked)} results")
            return reranked

        except Exception as e:
            logger.warning(f"Reranking failed: {e}, returning original results")
            return candidates[:top_k]

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_by: Optional[Dict[str, str]] = None,
        use_rerank: Optional[bool] = None,
    ) -> SemanticSearchResult:
        """
        Search for funds using natural language query.

        Parameters
        ----------
        query : str
            Natural language search query in Portuguese or English
        top_k : int
            Number of results to return (default 10)
        filter_by : dict, optional
            Metadata filters, e.g., {'investment_class': 'Ações'}
        use_rerank : bool, optional
            Override default rerank setting for this query

        Returns
        -------
        SemanticSearchResult
            Search result with matches and metadata

        Examples
        --------
        >>> tool = SemanticSearchTool()
        >>> result = tool.search("Bradesco gold fund", top_k=5)
        >>> if result.success:
        ...     for match in result.matches:
        ...         print(f"{match.cnpj}: {match.legal_name} (score: {match.score:.3f})")
        """
        start_time = datetime.now()

        # Validate inputs using Pydantic
        try:
            search_query = SemanticSearchQuery(
                query=query,
                top_k=top_k,
                filter_by=filter_by,
                use_rerank=use_rerank if use_rerank is not None else self.use_rerank
            )
        except Exception as e:
            return SemanticSearchResult(
                success=False,
                query=query,
                error_message=f"Invalid query parameters: {e}"
            )

        if self.index is None:
            return SemanticSearchResult(
                success=False,
                query=query,
                error_message="Index not initialized. Call build_index() first."
            )

        try:
            # Generate query embedding
            logger.debug(f"Generating embedding for query: {query}")
            query_embedding = self._generate_embedding(query)

            # Search in Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=search_query.top_k * 2 if search_query.use_rerank else search_query.top_k,
                filter=search_query.filter_by,
                include_metadata=True
            )

            # Convert to SemanticSearchMatch objects
            matches = []
            for match in results.matches:
                search_match = SemanticSearchMatch(
                    cnpj=match.id,
                    score=match.score,
                    legal_name=match.metadata.get('legal_name', ''),
                    trade_name=match.metadata.get('trade_name', ''),
                    investment_class=match.metadata.get('investment_class', ''),
                    anbima_classification=match.metadata.get('anbima_classification', ''),
                    fund_type=match.metadata.get('fund_type', ''),
                    structure=match.metadata.get('structure', ''),
                    target_audience=match.metadata.get('target_audience', ''),
                    data_quality=match.metadata.get('data_quality', ''),
                    matched_text_preview=match.metadata.get('searchable_text_preview', ''),
                )
                matches.append(search_match)

            # Apply reranking if enabled
            reranked = False
            if search_query.use_rerank and len(matches) > search_query.top_k:
                matches = self._rerank_results(query, matches, search_query.top_k)
                reranked = True
            else:
                matches = matches[:search_query.top_k]

            execution_time = (datetime.now() - start_time).total_seconds() * 1000

            return SemanticSearchResult(
                success=True,
                matches=matches,
                total_matches=len(matches),
                query=query,
                embedding_model=self.embedding_model,
                vector_dimension=self.dimension,
                reranked=reranked,
                execution_time_ms=execution_time
            )

        except Exception as e:
            logger.error(f"Error during search: {e}")
            execution_time = (datetime.now() - start_time).total_seconds() * 1000
            return SemanticSearchResult(
                success=False,
                query=query,
                embedding_model=self.embedding_model,
                vector_dimension=self.dimension,
                execution_time_ms=execution_time,
                error_message=str(e)
            )

    def search_with_explanation(
        self,
        query: str,
        top_k: int = 5,
    ) -> str:
        """
        Search and return results with human-readable explanation.

        Parameters
        ----------
        query : str
            Natural language search query
        top_k : int
            Number of results to return

        Returns
        -------
        str
            Human-readable explanation with search results
        """
        result = self.search(query, top_k=top_k)

        if not result.success:
            return f"Search failed: {result.error_message}"

        explanation = f"""
            Semantic Search Results for: "{query}"

            Found {result.total_matches} matching funds using vector similarity search.
            Embedding Model: OpenAI {result.embedding_model} ({result.vector_dimension}D)
            Vector Database: Pinecone (Serverless)
            Reranked: {'Yes' if result.reranked else 'No'}
            Execution Time: {result.execution_time_ms:.0f}ms

            Top matches ranked by semantic similarity:
            """

        for idx, match in enumerate(result.matches, 1):
            score_info = f"Similarity: {match.score:.1%}"
            if match.rerank_score is not None:
                score_info += f" | Rerank: {match.rerank_score:.1%}"

            explanation += f"""
                {idx}. {match.legal_name}
                CNPJ: {match.cnpj}
                Type: {match.investment_class} | {match.anbima_classification}
                {score_info}
                Quality: {match.data_quality}
                """

        return explanation

    def get_index_stats(self) -> IndexStats:
        """
        Get Pinecone index statistics.

        Returns
        -------
        IndexStats
            Index statistics with vector counts and configuration
        """
        if self.index is None:
            raise RuntimeError("Index not initialized. Call build_index() first.")

        stats = self.index.describe_index_stats()
        return IndexStats(
            total_vectors=stats.total_vector_count,
            dimension=self.dimension,
            index_fullness=stats.index_fullness,
            metric=self.metric,
            index_name=self.index_name,
        )


if __name__ == "__main__":
    # Demo usage
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    print("=" * 80)
    print("SEMANTIC SEARCH TOOL - PINECONE + OPENAI DEMO")
    print("=" * 80)

    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        exit(1)

    if not os.getenv("PINECONE_API_KEY"):
        print("Error: PINECONE_API_KEY environment variable not set")
        print("Set it with: export PINECONE_API_KEY='your-key-here'")
        exit(1)

    # Initialize tool
    tool = SemanticSearchTool()

    # Build index
    print("\n📦 Building vector index...")
    build_result = tool.build_index()

    if build_result.success:
        print(f"Indexed {build_result.total_vectors:,} funds")
        print(f"Rebuild: {build_result.rebuild}")
        print(f"Time: {build_result.execution_time_ms:.0f}ms")
    else:
        print(f"Index build failed: {build_result.error_message}")
        exit(1)

    # Show index stats
    stats = tool.get_index_stats()
    print(f"\nIndex Statistics:")
    print(f"   Index: {stats.index_name}")
    print(f"   Total vectors: {stats.total_vectors:,}")
    print(f"   Dimension: {stats.dimension}")
    print(f"   Metric: {stats.metric}")
    print(f"   Index fullness: {stats.index_fullness:.2%}")

    # Example searches
    test_queries = [
        "Bradesco gold fund",
        "fundos de renda fixa conservadores",
        "sustainable technology investing",
        "Latin American equity exposure",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        explanation = tool.search_with_explanation(query, top_k=3)
        print(explanation)
