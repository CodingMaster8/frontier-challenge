"""Run comprehensive evaluations for all agent tools."""

import asyncio
import logging
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evals.evaluator import AgentEvaluator
from evals.test_cases import get_all_suites, get_edge_cases
from evals.models import ToolType

# Import tool executors
from evals.run_semantic_eval import execute_semantic_search
from evals.run_sql_eval import execute_structured_filter
from evals.run_holdings_eval import execute_holdings_search

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def main():
    """Run comprehensive evaluation across all tools."""
    print("=" * 80)
    print("COMPREHENSIVE AGENT EVALUATION")
    print("=" * 80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize evaluator
    evaluator = AgentEvaluator(
        db_path="data/br_funds.db",
        output_dir="evals/results"
    )

    # Get all test suites
    all_suites = get_all_suites()

    # Map tool types to executors
    executors = {
        ToolType.SEMANTIC_SEARCH: execute_semantic_search,
        ToolType.STRUCTURED_FILTER: execute_structured_filter,
        ToolType.HOLDINGS_SEARCH: execute_holdings_search,
    }

    # Track results by tool
    tool_results = {}
    all_results = []

    # Run evaluations for each tool
    for tool_type, suite in all_suites.items():
        # Skip agent routing for now (requires different executor)
        if tool_type == ToolType.AGENT_ROUTING:
            continue

        print("\n" + "=" * 80)
        print(f"EVALUATING: {suite.name}")
        print("=" * 80)
        print(f"Cases: {len(suite.cases)}")

        executor = executors.get(tool_type)
        if not executor:
            logger.warning(f"No executor found for {tool_type}")
            continue

        # Run evaluation
        results = await evaluator.evaluate_suite(
            suite=suite,
            tool_executor=executor,
            parallel=False,
        )

        tool_results[tool_type] = results
        all_results.extend(results)

        # Print quick summary
        summary = evaluator.generate_summary(results)
        print(f"\n✓ Completed: {summary.passed}/{summary.total_cases} passed ({summary.pass_rate:.1%})")
        print(f"  Avg latency: {summary.avg_time_ms:.2f}ms")

    # Generate comprehensive summary
    print("\n" + "=" * 80)
    print("COMPREHENSIVE EVALUATION SUMMARY")
    print("=" * 80)

    overall_summary = evaluator.generate_summary(all_results)

    print(f"\nTotal Cases: {overall_summary.total_cases}")
    print(f"Passed: {overall_summary.passed} ({overall_summary.pass_rate:.1%})")
    print(f"Failed: {overall_summary.failed}")
    print(f"Errors: {overall_summary.errors}")
    print(f"Error Rate: {overall_summary.error_rate:.1%}")
    print(f"\nTotal Time: {overall_summary.total_time_ms:.2f}ms")
    print(f"Avg Time per Case: {overall_summary.avg_time_ms:.2f}ms")
    print(f"Overall Accuracy: {overall_summary.accuracy:.2%}")

    # Per-tool breakdown
    print("\n" + "-" * 80)
    print("PER-TOOL BREAKDOWN")
    print("-" * 80)

    for tool_type, results in tool_results.items():
        if not results:
            continue

        tool_report = evaluator.generate_tool_report(tool_type, results)

        print(f"\n{tool_type.value.upper()}")
        print(f"  Cases: {tool_report.total_cases}")
        print(f"  Pass Rate: {tool_report.passed}/{tool_report.total_cases} ({tool_report.passed/tool_report.total_cases:.1%})")
        print(f"  Accuracy: {tool_report.accuracy:.2%}")
        print(f"  Precision: {tool_report.precision:.2%}")
        print(f"  Recall: {tool_report.recall:.2%}")
        print(f"  F1 Score: {tool_report.f1_score:.2%}")
        print(f"  Avg Latency: {tool_report.avg_latency_ms:.2f}ms")

        if tool_report.failed_cases:
            print(f"  Failed Cases: {', '.join(tool_report.failed_cases[:5])}")
            if len(tool_report.failed_cases) > 5:
                print(f"    ... and {len(tool_report.failed_cases) - 5} more")

    # Edge case analysis
    print("\n" + "-" * 80)
    print("EDGE CASE ANALYSIS")
    print("-" * 80)

    edge_case_ids = {case.id for case in get_edge_cases()}
    edge_results = [r for r in all_results if r.case_id in edge_case_ids]

    if edge_results:
        edge_summary = evaluator.generate_summary(edge_results)
        print(f"Edge Cases: {edge_summary.total_cases}")
        print(f"Pass Rate: {edge_summary.passed}/{edge_summary.total_cases} ({edge_summary.pass_rate:.1%})")
        print(f"Error Rate: {edge_summary.error_rate:.1%}")
        print(f"Avg Accuracy: {edge_summary.accuracy:.2%}")

    # Overall metrics
    print("\n" + "-" * 80)
    print("OVERALL METRICS")
    print("-" * 80)

    if "overall" in overall_summary.tool_metrics:
        metrics = overall_summary.tool_metrics["overall"]
        for metric_name, value in sorted(metrics.items()):
            print(f"{metric_name:20s}: {value:.4f}")

    # Export all results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = evaluator.export_results(
        all_results,
        f"comprehensive_eval_{timestamp}.json"
    )
    print(f"\n✓ Results exported to: {output_file}")

    # Generate HTML report
    report_file = generate_html_report(
        evaluator,
        all_results,
        tool_results,
        overall_summary
    )
    print(f"✓ HTML report generated: {report_file}")

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    # Define thresholds
    PASS_RATE_THRESHOLD = 0.75
    ERROR_RATE_THRESHOLD = 0.15
    ACCURACY_THRESHOLD = 0.70

    passed = True

    if overall_summary.pass_rate < PASS_RATE_THRESHOLD:
        print(f"✗ Pass rate {overall_summary.pass_rate:.1%} is below {PASS_RATE_THRESHOLD:.0%} threshold")
        passed = False
    else:
        print(f"✓ Pass rate {overall_summary.pass_rate:.1%} meets threshold")

    if overall_summary.error_rate > ERROR_RATE_THRESHOLD:
        print(f"✗ Error rate {overall_summary.error_rate:.1%} exceeds {ERROR_RATE_THRESHOLD:.0%} threshold")
        passed = False
    else:
        print(f"✓ Error rate {overall_summary.error_rate:.1%} is acceptable")

    if overall_summary.accuracy < ACCURACY_THRESHOLD:
        print(f"✗ Accuracy {overall_summary.accuracy:.1%} is below {ACCURACY_THRESHOLD:.0%} threshold")
        passed = False
    else:
        print(f"✓ Accuracy {overall_summary.accuracy:.1%} meets threshold")

    print("\n" + "=" * 80)
    if passed:
        print("✓ COMPREHENSIVE EVALUATION PASSED")
        print("=" * 80)
        return 0
    else:
        print("✗ COMPREHENSIVE EVALUATION FAILED")
        print("=" * 80)
        return 1


def generate_html_report(evaluator, all_results, tool_results, summary):
    """Generate an HTML report of the evaluation results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = evaluator.output_dir / f"eval_report_{timestamp}.html"

    html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Agent Evaluation Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
        h2 {{ color: #34495e; margin-top: 30px; }}
        .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; }}
        .metric-card.success {{ background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }}
        .metric-card.warning {{ background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }}
        .metric-card.error {{ background: linear-gradient(135deg, #fa709a 0%, #fee140 100%); }}
        .metric-value {{ font-size: 2em; font-weight: bold; }}
        .metric-label {{ font-size: 0.9em; opacity: 0.9; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #3498db; color: white; }}
        tr:hover {{ background-color: #f5f5f5; }}
        .status-passed {{ color: #27ae60; font-weight: bold; }}
        .status-failed {{ color: #e74c3c; font-weight: bold; }}
        .status-error {{ color: #e67e22; font-weight: bold; }}
        .progress-bar {{ width: 100%; height: 30px; background-color: #ecf0f1; border-radius: 15px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%); transition: width 0.3s; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 Agent Evaluation Report</h1>
        <p><strong>Generated:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <h2>📊 Overall Summary</h2>
        <div class="summary">
            <div class="metric-card success">
                <div class="metric-value">{summary.passed}</div>
                <div class="metric-label">Passed Cases</div>
            </div>
            <div class="metric-card warning">
                <div class="metric-value">{summary.failed}</div>
                <div class="metric-label">Failed Cases</div>
            </div>
            <div class="metric-card error">
                <div class="metric-value">{summary.errors}</div>
                <div class="metric-label">Errors</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.pass_rate:.1%}</div>
                <div class="metric-label">Pass Rate</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.accuracy:.1%}</div>
                <div class="metric-label">Accuracy</div>
            </div>
            <div class="metric-card">
                <div class="metric-value">{summary.avg_time_ms:.0f}ms</div>
                <div class="metric-label">Avg Latency</div>
            </div>
        </div>

        <h2>🔧 Per-Tool Results</h2>
        <table>
            <thead>
                <tr>
                    <th>Tool</th>
                    <th>Total Cases</th>
                    <th>Pass Rate</th>
                    <th>Accuracy</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>F1 Score</th>
                    <th>Avg Latency</th>
                </tr>
            </thead>
            <tbody>
"""

    for tool_type, results in tool_results.items():
        if not results:
            continue
        tool_report = evaluator.generate_tool_report(tool_type, results)
        pass_rate = tool_report.passed / tool_report.total_cases if tool_report.total_cases > 0 else 0

        html_content += f"""
                <tr>
                    <td><strong>{tool_type.value}</strong></td>
                    <td>{tool_report.total_cases}</td>
                    <td>{pass_rate:.1%}</td>
                    <td>{tool_report.accuracy:.2%}</td>
                    <td>{tool_report.precision:.2%}</td>
                    <td>{tool_report.recall:.2%}</td>
                    <td>{tool_report.f1_score:.2%}</td>
                    <td>{tool_report.avg_latency_ms:.2f}ms</td>
                </tr>
"""

    html_content += """
            </tbody>
        </table>

        <h2>📈 Metrics Breakdown</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Value</th>
                </tr>
            </thead>
            <tbody>
"""

    if "overall" in summary.tool_metrics:
        for metric_name, value in sorted(summary.tool_metrics["overall"].items()):
            html_content += f"""
                <tr>
                    <td>{metric_name}</td>
                    <td>{value:.4f}</td>
                </tr>
"""

    html_content += """
            </tbody>
        </table>
    </div>
</body>
</html>
"""

    with open(report_path, 'w') as f:
        f.write(html_content)

    return report_path


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
