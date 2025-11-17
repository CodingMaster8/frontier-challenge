"""Run evaluations for the Structured Filter Tool (SQL)."""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from frontier_challenge.tools import StructuredFilterTool
from evals.evaluator import AgentEvaluator
from evals.test_cases import get_suite_by_tool
from evals.models import ToolType, EvalCase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def execute_structured_filter(case: EvalCase):
    """Execute structured filter tool for a test case."""
    tool = StructuredFilterTool(db_path="data/br_funds.db", refine_query=True)

    try:
        # Execute filter (async method)
        result = await tool.structured_filter(query=case.input_query)

        # Convert result to dict format
        return {
            "success": result.success,
            "funds": [
                {
                    "cnpj": fund.cnpj,
                    "legal_name": fund.legal_name,
                    "return_ytd_2024_avg": fund.return_ytd_2024_avg,
                    "management_fee_pct": fund.management_fee_pct,
                    "nav": fund.nav,
                }
                for fund in result.funds
            ] if result.funds else [],
            "total_count": result.total_count,
            "sql_query": result.sql_query,
            "execution_time_ms": result.execution_time_ms,
            "error_message": result.error_message,
        }
    except Exception as e:
        logger.error(f"Error executing filter: {e}", exc_info=True)
        return {
            "success": False,
            "error_message": str(e),
            "funds": [],
        }


async def main():
    """Main evaluation runner."""
    print("=" * 80)
    print("STRUCTURED FILTER TOOL (SQL) - EVALUATION")
    print("=" * 80)

    # Initialize evaluator
    evaluator = AgentEvaluator(
        db_path="data/br_funds.db",
        output_dir="evals/results"
    )

    # Load test suite
    suite = get_suite_by_tool(ToolType.STRUCTURED_FILTER)
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
        tool_executor=execute_structured_filter,
        parallel=False,
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
        tool_type=ToolType.STRUCTURED_FILTER,
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
    output_file = evaluator.export_results(results, "structured_filter_eval.json")
    print(f"\n✓ Results exported to: {output_file}")

    # Print pass/fail status
    print("\n" + "=" * 80)
    if summary.pass_rate >= 0.75:  # 75% pass rate threshold (more lenient for SQL)
        print("✓ EVALUATION PASSED")
        return 0
    else:
        print("✗ EVALUATION FAILED")
        print(f"  Pass rate {summary.pass_rate:.1%} is below 75% threshold")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
