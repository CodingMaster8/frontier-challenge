"""Run evaluations for the Holdings Search Tool."""

import asyncio
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from frontier_challenge.tools import HoldingsSearchTool
from evals.evaluator import AgentEvaluator
from evals.test_cases import get_suite_by_tool
from evals.models import ToolType, EvalCase

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def execute_holdings_search(case: EvalCase):
    """Execute holdings search tool for a test case."""
    tool = HoldingsSearchTool(db_path="data/br_funds.db")

    try:
        # Execute search using query
        result = await tool.search_holdings(
            query=case.input_query
        )

        # Convert result to dict format
        output = {
            "success": result.success,
            "execution_time_ms": result.execution_time_ms,
            "error_message": result.error_message,
            "search_method": result.search_method,
            "total_count": result.total_count,
            "unique_funds_count": result.unique_funds_count,
        }

        # Use fund_summaries if available, otherwise use holdings
        if result.fund_summaries:
            output["holdings"] = [
                {
                    "cnpj": summary.cnpj,
                    "legal_name": summary.legal_name,
                    "asset_name": summary.asset_name,
                    "portfolio_weight_pct": summary.portfolio_weight_pct,
                    "company_name": summary.asset_name,
                }
                for summary in result.fund_summaries
            ]
        elif result.holdings:
            output["holdings"] = [
                {
                    "cnpj": holding.cnpj,
                    "legal_name": holding.legal_name,
                    "asset_name": holding.asset_name,
                    "portfolio_weight_pct": holding.portfolio_weight_pct,
                    "company_name": holding.asset_name,
                }
                for holding in result.holdings
            ]
        else:
            output["holdings"] = []

        return output

    except Exception as e:
        logger.error(f"Error executing holdings search: {e}", exc_info=True)
        return {
            "success": False,
            "error_message": str(e),
            "holdings": [],
        }


async def main():
    """Main evaluation runner."""
    print("=" * 80)
    print("HOLDINGS SEARCH TOOL - EVALUATION")
    print("=" * 80)

    # Initialize evaluator
    evaluator = AgentEvaluator(
        db_path="data/br_funds.db",
        output_dir="evals/results"
    )

    # Load test suite
    suite = get_suite_by_tool(ToolType.HOLDINGS_SEARCH)
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
        tool_executor=execute_holdings_search,
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
        tool_type=ToolType.HOLDINGS_SEARCH,
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
    output_file = evaluator.export_results(results, "holdings_search_eval.json")
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
