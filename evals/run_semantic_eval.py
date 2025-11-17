"""Run evaluations for the Semantic Search Tool."""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from frontier_challenge.tools import SemanticSearchTool
from evals.evaluator import AgentEvaluator
from evals.test_cases import get_suite_by_tool
from evals.models import ToolType, EvalCase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def execute_semantic_search(case: EvalCase):
    """Execute semantic search tool for a test case."""
    tool = SemanticSearchTool(db_path="data/br_funds.db")

    # Build index if needed
    try:
        build_result = tool.build_index(force_rebuild=False)
        if not build_result.success:
            logger.error(f"Failed to build index: {build_result.error_message}")
            return {"success": False, "error_message": build_result.error_message}
    except Exception as e:
        logger.error(f"Error building index: {e}")
        return {"success": False, "error_message": str(e)}

    # Execute search
    try:
        top_k = case.metadata.get("top_k", 10)
        result = tool.semantic_search(
            query=case.input_query,
            top_k=top_k
        )

        # Convert result to dict format
        return {
            "success": result.success,
            "funds": [
                {
                    "cnpj": match.cnpj,
                    "legal_name": match.legal_name,
                    "similarity_score": match.score,
                }
                for match in result.matches
            ] if result.matches else [],
            "total_count": result.total_matches,
            "execution_time_ms": result.execution_time_ms,
            "error_message": result.error_message,
            "relevance_scores": [match.score for match in result.matches] if result.matches else [],
        }
    except Exception as e:
        logger.error(f"Error executing search: {e}", exc_info=True)
        return {
            "success": False,
            "error_message": str(e),
            "funds": [],
        }


async def main():
    """Main evaluation runner."""
    print("=" * 80)
    print("SEMANTIC SEARCH TOOL - EVALUATION")
    print("=" * 80)

    # Initialize evaluator
    evaluator = AgentEvaluator(
        db_path="data/br_funds.db",
        output_dir="evals/results"
    )

    # Load test suite
    suite = get_suite_by_tool(ToolType.SEMANTIC_SEARCH)
    print(f"\nLoaded test suite: {suite.name}")
    print(f"Description: {suite.description}")
    print(f"Total cases: {len(suite.cases)}")

    # Count edge cases
    edge_cases = [c for c in suite.cases if c.edge_case]
    print(f"Edge cases: {len(edge_cases)}")

    # Run evaluation
    print("\n" + "=" * 80)
    print("RUNNING EVALUATIONS")
    print("=" * 80)

    results = await evaluator.evaluate_suite(
        suite=suite,
        tool_executor=execute_semantic_search,
        parallel=False,  # Run sequentially for better logging
    )

    # Generate summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    summary = evaluator.generate_summary(results)

    print(f"\nTotal Cases: {summary.total_cases}")
    print(f"Passed: {summary.passed} ({summary.pass_rate:.1%})")
    print(f"Failed: {summary.failed}")
    print(f"Errors: {summary.errors}")
    print(f"Error Rate: {summary.error_rate:.1%}")
    print(f"\nTotal Time: {summary.total_time_ms:.2f}ms")
    print(f"Avg Time per Case: {summary.avg_time_ms:.2f}ms")
    print(f"Overall Accuracy: {summary.accuracy:.2%}")

    # Print detailed metrics
    print("\n" + "-" * 80)
    print("DETAILED METRICS")
    print("-" * 80)

    if "overall" in summary.tool_metrics:
        metrics = summary.tool_metrics["overall"]
        for metric_name, value in sorted(metrics.items()):
            print(f"{metric_name:20s}: {value:.4f}")

    # Generate tool report
    tool_report = evaluator.generate_tool_report(
        tool_type=ToolType.SEMANTIC_SEARCH,
        results=results
    )

    print("\n" + "-" * 80)
    print("TOOL-SPECIFIC REPORT")
    print("-" * 80)
    print(f"Accuracy: {tool_report.accuracy:.2%}")
    print(f"Precision: {tool_report.precision:.2%}")
    print(f"Recall: {tool_report.recall:.2%}")
    print(f"F1 Score: {tool_report.f1_score:.2%}")
    print(f"Avg Latency: {tool_report.avg_latency_ms:.2f}ms")

    # Show failed cases
    if tool_report.failed_cases:
        print("\n" + "-" * 80)
        print("FAILED CASES")
        print("-" * 80)
        for case_id in tool_report.failed_cases:
            case = next((c for c in suite.cases if c.id == case_id), None)
            if case:
                print(f"- {case_id}: {case.name}")
                result = next((r for r in results if r.case_id == case_id), None)
                if result and result.error_message:
                    print(f"  Error: {result.error_message}")

    # Export results
    output_file = evaluator.export_results(results, "semantic_search_eval.json")
    print(f"\n✓ Results exported to: {output_file}")

    # Print pass/fail status
    print("\n" + "=" * 80)
    if summary.pass_rate >= 0.8:  # 80% pass rate threshold
        print("✓ EVALUATION PASSED")
        return 0
    else:
        print("✗ EVALUATION FAILED")
        print(f"  Pass rate {summary.pass_rate:.1%} is below 80% threshold")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
