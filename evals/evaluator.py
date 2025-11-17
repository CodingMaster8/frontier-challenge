"""Core evaluation framework for the Financial Agent."""

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Callable

from .models import (
    EvalCase,
    EvalResult,
    EvalStatus,
    EvalSummary,
    ToolEvalResult,
    ToolType,
)
from .metrics import (
    AccuracyMetric,
    PrecisionMetric,
    RecallMetric,
    F1Metric,
    LatencyMetric,
    ErrorRateMetric,
    RelevanceMetric,
    CompletenessMetric,
    RobustnessMetric,
)

logger = logging.getLogger(__name__)


class EvalSuite:
    """Suite of evaluation cases for a specific tool or the entire agent."""

    def __init__(
        self,
        name: str,
        description: str,
        cases: List[EvalCase],
        tool_type: Optional[ToolType] = None,
    ):
        self.name = name
        self.description = description
        self.cases = cases
        self.tool_type = tool_type

    def filter_by_tags(self, tags: List[str]) -> "EvalSuite":
        """Filter cases by tags."""
        filtered = [
            case for case in self.cases
            if any(tag in case.tags for tag in tags)
        ]
        return EvalSuite(
            name=f"{self.name} (filtered)",
            description=f"{self.description} - filtered by tags: {tags}",
            cases=filtered,
            tool_type=self.tool_type,
        )

    def filter_edge_cases(self) -> "EvalSuite":
        """Get only edge cases."""
        filtered = [case for case in self.cases if case.edge_case]
        return EvalSuite(
            name=f"{self.name} (edge cases)",
            description=f"{self.description} - edge cases only",
            cases=filtered,
            tool_type=self.tool_type,
        )


class AgentEvaluator:
    """Main evaluator for the Financial Agent and its tools."""

    def __init__(
        self,
        db_path: str = "data/br_funds.db",
        output_dir: str = "evals/results",
    ):
        self.db_path = db_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize metrics
        self.metrics = [
            AccuracyMetric(),
            PrecisionMetric(),
            RecallMetric(),
            F1Metric(),
            LatencyMetric(),
            ErrorRateMetric(),
            RelevanceMetric(),
            CompletenessMetric(),
            RobustnessMetric(),
        ]

        self.results: List[EvalResult] = []

    async def evaluate_case(
        self,
        case: EvalCase,
        tool_executor: Callable,
    ) -> EvalResult:
        """
        Evaluate a single test case.

        Parameters
        ----------
        case : EvalCase
            The test case to evaluate
        tool_executor : Callable
            Function to execute the tool with the input

        Returns
        -------
        EvalResult
            The evaluation result
        """
        logger.info(f"Evaluating case: {case.id} - {case.name}")

        start_time = time.time()

        try:
            # Execute the tool
            actual_output = await tool_executor(case)

            execution_time_ms = (time.time() - start_time) * 1000

            # Check if this is an expected failure
            if case.should_fail:
                if actual_output.get("success", False):
                    status = EvalStatus.FAILED
                    error_message = "Expected failure but tool succeeded"
                else:
                    status = EvalStatus.PASSED
                    error_message = None
            else:
                # Run checks
                passed_checks, failed_checks = self._run_checks(case, actual_output)

                status = EvalStatus.PASSED if len(failed_checks) == 0 else EvalStatus.FAILED
                error_message = ", ".join(failed_checks) if failed_checks else None

            # Compute metrics
            metrics_dict = self._compute_metrics(case, actual_output, execution_time_ms)

            result = EvalResult(
                case_id=case.id,
                status=status,
                actual_output=actual_output,
                expected_output=case.expected_output,
                error_message=error_message,
                execution_time_ms=execution_time_ms,
                metrics=metrics_dict,
                passed_checks=passed_checks if not case.should_fail else ["Expected failure occurred"],
                failed_checks=failed_checks if not case.should_fail else [],
            )

        except Exception as e:
            logger.error(f"Error evaluating case {case.id}: {e}", exc_info=True)

            execution_time_ms = (time.time() - start_time) * 1000

            result = EvalResult(
                case_id=case.id,
                status=EvalStatus.ERROR,
                error_message=str(e),
                execution_time_ms=execution_time_ms,
                failed_checks=[f"Exception: {str(e)}"],
            )

        return result

    def _run_checks(
        self,
        case: EvalCase,
        actual_output: Dict[str, Any]
    ) -> tuple[List[str], List[str]]:
        """Run validation checks on the output."""
        passed = []
        failed = []

        # Check success
        if not actual_output.get("success", False):
            failed.append("Tool execution failed")
            return passed, failed

        passed.append("Tool execution succeeded")

        # Check result count
        results = actual_output.get("funds") or actual_output.get("holdings") or []
        if case.min_results is not None:
            if len(results) >= case.min_results:
                passed.append(f"Min results check passed ({len(results)} >= {case.min_results})")
            else:
                failed.append(f"Min results check failed ({len(results)} < {case.min_results})")

        if case.max_results is not None:
            if len(results) <= case.max_results:
                passed.append(f"Max results check passed ({len(results)} <= {case.max_results})")
            else:
                failed.append(f"Max results check failed ({len(results)} > {case.max_results})")

        # Check expected funds
        if case.expected_funds:
            actual_cnpjs = [
                fund.get("cnpj") or fund.get("cnpj_fundo")
                for fund in results
                if isinstance(fund, dict)
            ]
            found = sum(1 for cnpj in case.expected_funds if cnpj in actual_cnpjs)
            if found == len(case.expected_funds):
                passed.append(f"All expected funds found ({found}/{len(case.expected_funds)})")
            else:
                failed.append(f"Not all expected funds found ({found}/{len(case.expected_funds)})")

        # Check expected companies (for holdings search)
        if case.expected_companies:
            actual_companies = [
                holding.get("asset_name") or holding.get("company_name")
                for holding in results
                if isinstance(holding, dict)
            ]
            found = sum(
                1 for company in case.expected_companies
                if any(company.lower() in str(ac).lower() for ac in actual_companies)
            )
            if found == len(case.expected_companies):
                passed.append(f"All expected companies found ({found}/{len(case.expected_companies)})")
            else:
                failed.append(f"Not all expected companies found ({found}/{len(case.expected_companies)})")

        # Check SQL pattern (for structured filter)
        if case.expected_sql_pattern:
            sql_query = actual_output.get("sql_query", "")
            if case.expected_sql_pattern.lower() in sql_query.lower():
                passed.append(f"SQL pattern found: {case.expected_sql_pattern}")
            else:
                failed.append(f"SQL pattern not found: {case.expected_sql_pattern}")

        # Check visualization type
        if case.expected_visualization_type:
            viz_type = actual_output.get("visualization_type")
            if viz_type == case.expected_visualization_type:
                passed.append(f"Correct visualization type: {viz_type}")
            else:
                failed.append(f"Wrong visualization type: expected {case.expected_visualization_type}, got {viz_type}")

        return passed, failed

    def _compute_metrics(
        self,
        case: EvalCase,
        actual_output: Dict[str, Any],
        execution_time_ms: float,
    ) -> Dict[str, float]:
        """Compute all metrics for the output."""
        metrics_dict = {}

        # Extract actual and expected values based on tool type
        actual_value = self._extract_value(actual_output, case.tool_type)
        expected_value = self._extract_expected_value(case)

        metadata = {
            "execution_time_ms": execution_time_ms,
            "success": actual_output.get("success", False),
            "error_handled": actual_output.get("error_message") is not None,
        }

        # Compute each metric
        for metric in self.metrics:
            try:
                value = metric.compute(actual_value, expected_value, metadata)
                metrics_dict[metric.name] = value
            except Exception as e:
                logger.warning(f"Error computing metric {metric.name}: {e}")
                metrics_dict[metric.name] = 0.0

        return metrics_dict

    def _extract_value(self, output: Dict[str, Any], tool_type: ToolType) -> Any:
        """Extract the relevant value from tool output."""
        if tool_type in [ToolType.SEMANTIC_SEARCH, ToolType.STRUCTURED_FILTER]:
            funds = output.get("funds", [])
            return [f.get("cnpj") or f.get("cnpj_fundo") for f in funds if isinstance(f, dict)]

        elif tool_type == ToolType.HOLDINGS_SEARCH:
            holdings = output.get("holdings", [])
            return [h.get("asset_name") for h in holdings if isinstance(h, dict)]

        elif tool_type == ToolType.VISUALIZATION:
            return output.get("visualization_type")

        return output

    def _extract_expected_value(self, case: EvalCase) -> Any:
        """Extract expected value from test case."""
        if case.expected_funds:
            return case.expected_funds
        elif case.expected_companies:
            return case.expected_companies
        elif case.expected_visualization_type:
            return case.expected_visualization_type
        elif case.expected_output:
            return case.expected_output
        return None

    async def evaluate_suite(
        self,
        suite: EvalSuite,
        tool_executor: Callable,
        parallel: bool = False,
    ) -> List[EvalResult]:
        """
        Evaluate an entire suite of test cases.

        Parameters
        ----------
        suite : EvalSuite
            The test suite to evaluate
        tool_executor : Callable
            Function to execute the tool
        parallel : bool
            Whether to run cases in parallel

        Returns
        -------
        List[EvalResult]
            Results for all test cases
        """
        logger.info(f"Evaluating suite: {suite.name} ({len(suite.cases)} cases)")

        if parallel:
            tasks = [
                self.evaluate_case(case, tool_executor)
                for case in suite.cases
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
        else:
            results = []
            for case in suite.cases:
                result = await self.evaluate_case(case, tool_executor)
                results.append(result)

        self.results.extend(results)
        return results

    def generate_summary(
        self,
        results: Optional[List[EvalResult]] = None
    ) -> EvalSummary:
        """Generate summary statistics from evaluation results."""
        if results is None:
            results = self.results

        if not results:
            return EvalSummary(
                total_cases=0,
                passed=0,
                failed=0,
                errors=0,
                skipped=0,
                total_time_ms=0.0,
                avg_time_ms=0.0,
                accuracy=0.0,
            )

        passed = sum(1 for r in results if r.status == EvalStatus.PASSED)
        failed = sum(1 for r in results if r.status == EvalStatus.FAILED)
        errors = sum(1 for r in results if r.status == EvalStatus.ERROR)
        skipped = sum(1 for r in results if r.status == EvalStatus.SKIPPED)

        total_time = sum(r.execution_time_ms for r in results)
        avg_time = total_time / len(results) if results else 0.0

        # Calculate overall accuracy
        accuracy = sum(r.accuracy for r in results) / len(results) if results else 0.0

        # Calculate per-metric averages
        tool_metrics = {}
        metric_names = set()
        for result in results:
            metric_names.update(result.metrics.keys())

        for metric_name in metric_names:
            values = [r.metrics.get(metric_name, 0.0) for r in results]
            tool_metrics[metric_name] = sum(values) / len(values) if values else 0.0

        return EvalSummary(
            total_cases=len(results),
            passed=passed,
            failed=failed,
            errors=errors,
            skipped=skipped,
            total_time_ms=total_time,
            avg_time_ms=avg_time,
            accuracy=accuracy,
            tool_metrics={"overall": tool_metrics},
        )

    def generate_tool_report(
        self,
        tool_type: ToolType,
        results: Optional[List[EvalResult]] = None,
    ) -> ToolEvalResult:
        """Generate a detailed report for a specific tool."""
        if results is None:
            results = self.results

        # Filter results for this tool type
        # This requires cases to have tool_type info, which we track separately

        passed = sum(1 for r in results if r.status == EvalStatus.PASSED)
        failed = sum(1 for r in results if r.status == EvalStatus.FAILED)
        errors = sum(1 for r in results if r.status == EvalStatus.ERROR)

        avg_latency = (
            sum(r.execution_time_ms for r in results) / len(results)
            if results else 0.0
        )

        accuracy = sum(r.metrics.get("accuracy", 0.0) for r in results) / len(results) if results else 0.0
        precision = sum(r.metrics.get("precision", 0.0) for r in results) / len(results) if results else 0.0
        recall = sum(r.metrics.get("recall", 0.0) for r in results) / len(results) if results else 0.0
        f1 = sum(r.metrics.get("f1_score", 0.0) for r in results) / len(results) if results else 0.0

        # Edge case pass rate (tracked separately)
        edge_case_pass_rate = 0.0

        failed_cases = [r.case_id for r in results if r.status == EvalStatus.FAILED]

        # Aggregate all metrics
        all_metrics = {}
        for result in results:
            for metric_name, value in result.metrics.items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(value)

        avg_metrics = {
            name: sum(values) / len(values) if values else 0.0
            for name, values in all_metrics.items()
        }

        return ToolEvalResult(
            tool_type=tool_type,
            total_cases=len(results),
            passed=passed,
            failed=failed,
            errors=errors,
            avg_latency_ms=avg_latency,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            edge_case_pass_rate=edge_case_pass_rate,
            failed_cases=failed_cases,
            metrics=avg_metrics,
        )

    def export_results(
        self,
        results: Optional[List[EvalResult]] = None,
        filename: Optional[str] = None,
    ) -> Path:
        """Export results to JSON file."""
        import json
        from dataclasses import asdict

        if results is None:
            results = self.results

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_results_{timestamp}.json"

        output_path = self.output_dir / filename

        # Convert results to dict
        results_dict = [
            {
                **asdict(r),
                "timestamp": r.timestamp.isoformat(),
            }
            for r in results
        ]

        with open(output_path, "w") as f:
            json.dump(results_dict, f, indent=2, default=str)

        logger.info(f"Results exported to {output_path}")
        return output_path
