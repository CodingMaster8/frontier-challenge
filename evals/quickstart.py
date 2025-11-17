#!/usr/bin/env python3
"""
Quick start script for running evaluations.

Usage:
    python evals/quickstart.py                  # Run all evaluations
    python evals/quickstart.py --tool semantic  # Run specific tool
    python evals/quickstart.py --quick          # Run quick subset
    python evals/quickstart.py --edge-only      # Run only edge cases
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def main():
    parser = argparse.ArgumentParser(
        description="Quick start for agent evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          Run all evaluations
  %(prog)s --tool semantic          Run semantic search evaluation
  %(prog)s --tool sql               Run SQL filter evaluation
  %(prog)s --tool holdings          Run holdings search evaluation
  %(prog)s --quick                  Run quick subset of tests
  %(prog)s --edge-only              Run only edge cases
  %(prog)s --tags fuzzy misspelling Run tests with specific tags
        """
    )

    parser.add_argument(
        "--tool",
        choices=["semantic", "sql", "holdings", "all"],
        default="all",
        help="Which tool to evaluate (default: all)"
    )

    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick subset (basic cases only)"
    )

    parser.add_argument(
        "--edge-only",
        action="store_true",
        help="Run only edge cases"
    )

    parser.add_argument(
        "--tags",
        nargs="+",
        help="Run only tests with these tags"
    )

    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel (faster but less readable output)"
    )

    parser.add_argument(
        "--output-dir",
        default="evals/results",
        help="Output directory for results (default: evals/results)"
    )

    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML report generation"
    )

    args = parser.parse_args()

    # Print banner
    print("=" * 80)
    print("🤖 AGENT EVALUATION QUICK START")
    print("=" * 80)
    print()

    # Determine which script to run
    if args.tool == "semantic":
        print("Running: Semantic Search Tool Evaluation")
        from evals.run_semantic_eval import main as run_eval

    elif args.tool == "sql":
        print("Running: Structured Filter Tool (SQL) Evaluation")
        from evals.run_sql_eval import main as run_eval

    elif args.tool == "holdings":
        print("Running: Holdings Search Tool Evaluation")
        from evals.run_holdings_eval import main as run_eval

    else:  # all
        print("Running: Comprehensive Evaluation (All Tools)")
        from evals.run_all_evals import main as run_eval

    # Run the evaluation
    try:
        exit_code = asyncio.run(run_eval())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n\n⚠️  Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Error running evaluation: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
