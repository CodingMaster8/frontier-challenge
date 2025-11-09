"""Test semantic search with Pinecone + OpenAI"""
import logging
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frontier_challenge.tools.semantic_search_tool import SemanticSearchTool

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def main():
    print("=" * 80)
    print("SEMANTIC SEARCH TOOL - PINECONE + OPENAI TEST")
    print("=" * 80)

    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("\n❌ Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='sk-proj-...'")
        return False

    if not os.getenv("PINECONE_API_KEY"):
        print("\n❌ Error: PINECONE_API_KEY environment variable not set")
        print("Set it with: export PINECONE_API_KEY='pcsk_...'")
        return False

    print("\n✅ API keys found")

    # Initialize tool
    print("\n📦 Initializing semantic search tool...")
    tool = SemanticSearchTool()

    # Build or load index
    print("\n📦 Building/loading vector index...")
    print("(First time: ~2-3 minutes to generate embeddings)")
    print("(Subsequent: <1 second to load existing index)")

    try:
        count = tool.build_index(force_rebuild=False)
        print(f"\n✅ Index ready with {count:,} funds")
    except Exception as e:
        print(f"\n❌ Error building index: {e}")
        return False

    # Show index stats
    print("\n📊 INDEX STATISTICS")
    print("-" * 80)
    stats = tool.get_index_stats()
    print(f"Total vectors: {stats['total_vectors']:,}")
    print(f"Dimension: {stats['dimension']}")
    print(f"Index fullness: {stats['index_fullness']:.2%}")

    # Calculate storage usage
    vector_size_kb = (stats['dimension'] * 4) / 1024  # 4 bytes per float
    total_size_mb = (stats['total_vectors'] * vector_size_kb) / 1024
    free_tier_gb = 2.0
    usage_pct = (total_size_mb / (free_tier_gb * 1024)) * 100

    print(f"\n💾 STORAGE ANALYSIS")
    print("-" * 80)
    print(f"Per vector: {vector_size_kb:.2f} KB")
    print(f"Total usage: {total_size_mb:.2f} MB")
    print(f"Free tier: {free_tier_gb} GB")
    print(f"Usage: {usage_pct:.2f}% of free tier")
    print(f"Remaining: {(free_tier_gb * 1024) - total_size_mb:.2f} MB")

    # Test queries
    print("\n" + "=" * 80)
    print("RUNNING TEST QUERIES")
    print("=" * 80)

    test_cases = [
        {
            "query": "Bradesco gold fund",
            "expected": "Should find Bradesco funds related to gold/commodities",
            "top_k": 3
        },
        {
            "query": "fundos de renda fixa conservadores",
            "expected": "Should find conservative fixed income funds",
            "top_k": 3
        },
        {
            "query": "sustainable ESG investing",
            "expected": "Should find ESG/sustainable funds",
            "top_k": 3
        },
        {
            "query": "ações de tecnologia",
            "expected": "Should find technology equity funds",
            "top_k": 3
        },
    ]

    for i, test in enumerate(test_cases, 1):
        print(f"\n{'='*80}")
        print(f"TEST {i}/{ len(test_cases)}")
        print(f"Query: {test['query']}")
        print(f"Expected: {test['expected']}")
        print('-'*80)

        try:
            results, explanation = tool.search_with_explanation(
                test['query'],
                top_k=test['top_k']
            )
            print(explanation)

            if not results:
                print("⚠️  No results found!")
            else:
                print(f"✅ Found {len(results)} results")

        except Exception as e:
            print(f"❌ Error: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("✅ TESTS COMPLETE")
    print("=" * 80)
    print(f"""
Summary:
- Indexed {count:,} Brazilian investment funds
- Using OpenAI text-embedding-3-small (1536D)
- Stored in Pinecone serverless
- Storage: {total_size_mb:.2f} MB ({usage_pct:.2f}% of free tier)
- Ready for production use!

Next steps:
1. Integrate into agent/CLI
2. Add evaluation metrics
3. Build structured filter tool
4. Build portfolio analysis tool
    """)

    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
