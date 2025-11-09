"""
Semantic Search Tool for Brazilian Investment Funds - Pinecone Edition

This tool uses vector embeddings to enable natural language search across funds.
It supports Portuguese and English queries for fuzzy/conceptual fund discovery.

Uses:
- OpenAI text-embedding-3-small for embeddings (1536 dimensions)
- Pinecone for vector storage (free tier: 2GB, ~100K vectors)

Example queries:
- "Bradesco gold fund"
- "sustainable technology investing"
- "fundos de renda fixa conservadores"
- "Latin American equity exposure"
"""

import logging
import os
from typing import List, Dict, Optional, Tuple
import time

import duckdb
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

logger = logging.getLogger(__name__)


class SemanticSearchTool:
    """
    Semantic search tool using OpenAI embeddings and Pinecone vector DB.

    Requirements:
    - OPENAI_API_KEY environment variable
    - PINECONE_API_KEY environment variable
    """

    def __init__(
        self,
        db_path: str = "data/br_funds.db",
        index_name: str = "br-funds-search",
        embedding_model: str = "text-embedding-3-small",
        dimension: int = 1536,
        metric: str = "cosine",
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
        """
        self.db_path = db_path
        self.index_name = index_name
        self.embedding_model = embedding_model
        self.dimension = dimension
        self.metric = metric

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

    def _generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for a single text using OpenAI."""
        response = self.openai_client.embeddings.create(
            input=text,
            model=self.embedding_model
        )
        return response.data[0].embedding

    def _generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts (batch processing)."""
        # OpenAI allows up to 2048 texts per batch, we'll use 100 for safety
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

    def build_index(self, force_rebuild: bool = False) -> int:
        """
        Build or load the Pinecone vector index from fund data.

        Parameters
        ----------
        force_rebuild : bool
            If True, delete and rebuild index. Default False.

        Returns
        -------
        int
            Number of funds indexed
        """
        # Check if index exists
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.index_name in existing_indexes:
            if not force_rebuild:
                logger.info(f"Loading existing index: {self.index_name}")
                self.index = self.pc.Index(self.index_name)
                stats = self.index.describe_index_stats()
                count = stats.total_vector_count
                logger.info(f"Loaded index with {count:,} vectors")
                return count
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
        logger.info("✅ Index created and ready")

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

        logger.info(f"✅ Index built successfully! {total_count:,} funds indexed")
        logger.info(f"📊 Index stats: {stats}")

        return total_count

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_by: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
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

        Returns
        -------
        List[Dict]
            List of matching funds with metadata and similarity scores

        Examples
        --------
        >>> tool = SemanticSearchTool()
        >>> tool.build_index()
        >>> results = tool.search("Bradesco gold fund", top_k=5)
        >>> for r in results:
        ...     print(f"{r['cnpj']}: {r['legal_name']} (score: {r['score']:.3f})")
        """
        if self.index is None:
            raise RuntimeError("Index not initialized. Call build_index() first.")

        # Generate query embedding
        logger.debug(f"Generating embedding for query: {query}")
        query_embedding = self._generate_embedding(query)

        # Search in Pinecone
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            filter=filter_by,
            include_metadata=True
        )

        # Format results
        formatted_results = []
        for match in results.matches:
            result = {
                'cnpj': match.id,
                'score': match.score,
                'legal_name': match.metadata.get('legal_name', ''),
                'trade_name': match.metadata.get('trade_name', ''),
                'investment_class': match.metadata.get('investment_class', ''),
                'anbima_classification': match.metadata.get('anbima_classification', ''),
                'fund_type': match.metadata.get('fund_type', ''),
                'structure': match.metadata.get('structure', ''),
                'target_audience': match.metadata.get('target_audience', ''),
                'data_quality': match.metadata.get('data_quality', ''),
                'matched_text_preview': match.metadata.get('searchable_text_preview', ''),
            }
            formatted_results.append(result)

        return formatted_results

    def search_with_explanation(
        self,
        query: str,
        top_k: int = 5,
    ) -> Tuple[List[Dict], str]:
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
        results : List[Dict]
            Search results
        explanation : str
            Human-readable explanation of search process
        """
        results = self.search(query, top_k=top_k)

        explanation = f"""
🔍 Semantic Search Results for: "{query}"

Found {len(results)} matching funds using vector similarity search.
Embedding Model: OpenAI {self.embedding_model} ({self.dimension}D)
Vector Database: Pinecone (Serverless)

Top matches ranked by semantic similarity:
"""

        for idx, result in enumerate(results, 1):
            explanation += f"""
{idx}. {result['legal_name']}
   CNPJ: {result['cnpj']}
   Type: {result['investment_class']} | {result['anbima_classification']}
   Similarity: {result['score']:.1%}
   Quality: {result['data_quality']}
"""

        return results, explanation

    def get_index_stats(self) -> Dict:
        """Get Pinecone index statistics."""
        if self.index is None:
            raise RuntimeError("Index not initialized.")

        stats = self.index.describe_index_stats()
        return {
            'total_vectors': stats.total_vector_count,
            'dimension': self.dimension,
            'index_fullness': stats.index_fullness,
            'namespaces': stats.namespaces,
        }


# Convenience functions for quick use
def build_fund_index(
    db_path: str = "data/br_funds.db",
    force_rebuild: bool = False,
) -> int:
    """
    Build semantic search index (convenience function).

    Parameters
    ----------
    db_path : str
        Path to DuckDB database
    force_rebuild : bool
        Force rebuild even if index exists

    Returns
    -------
    int
        Number of funds indexed
    """
    tool = SemanticSearchTool(db_path=db_path)
    return tool.build_index(force_rebuild=force_rebuild)


def search_funds(
    query: str,
    top_k: int = 10,
    db_path: str = "data/br_funds.db",
) -> List[Dict]:
    """
    Search funds by natural language query (convenience function).

    Parameters
    ----------
    query : str
        Search query in Portuguese or English
    top_k : int
        Number of results to return
    db_path : str
        Path to DuckDB database

    Returns
    -------
    List[Dict]
        Matching funds with metadata
    """
    tool = SemanticSearchTool(db_path=db_path)
    tool.build_index()  # Load existing index
    return tool.search(query, top_k=top_k)


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
        print("❌ Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='your-key-here'")
        exit(1)

    if not os.getenv("PINECONE_API_KEY"):
        print("❌ Error: PINECONE_API_KEY environment variable not set")
        print("Set it with: export PINECONE_API_KEY='your-key-here'")
        exit(1)

    # Initialize tool
    tool = SemanticSearchTool()

    # Build index
    print("\n📦 Building vector index...")
    count = tool.build_index()
    print(f"✅ Indexed {count:,} funds")

    # Show index stats
    stats = tool.get_index_stats()
    print(f"\n📊 Index Statistics:")
    print(f"   Total vectors: {stats['total_vectors']:,}")
    print(f"   Dimension: {stats['dimension']}")
    print(f"   Index fullness: {stats['index_fullness']:.2%}")

    # Example searches
    test_queries = [
        "Bradesco gold fund",
        "fundos de renda fixa conservadores",
        "sustainable technology investing",
        "Latin American equity exposure",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        results, explanation = tool.search_with_explanation(query, top_k=3)
        print(explanation)
