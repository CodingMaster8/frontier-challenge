"""
Sample test to verify the evaluation framework.

This is a minimal test that doesn't require API keys or database.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from evals.models import EvalCase, EvalResult, EvalStatus, ToolType, EvalSummary
from evals.metrics import AccuracyMetric, PrecisionMetric, RecallMetric, F1Metric
from evals.evaluator import AgentEvaluator, EvalSuite


def test_models():
    """Test that models can be instantiated."""
    print("Testing models...")

    case = EvalCase(
        id="test_001",
        name="Test case",
        description="Test description",
        tool_type=ToolType.SEMANTIC_SEARCH,
        input_query="test query",
        tags=["test"],
    )

    assert case.id == "test_001"
    assert case.tool_type == ToolType.SEMANTIC_SEARCH
    print("✓ Models work correctly")


def test_metrics():
    """Test that metrics compute correctly."""
    print("\nTesting metrics...")

    # Test accuracy
    accuracy = AccuracyMetric()
    assert accuracy.compute(["a", "b"], ["a", "b"]) == 1.0
    assert accuracy.compute(["a"], ["a", "b"]) == 0.5
    print("✓ Accuracy metric works")

    # Test precision
    precision = PrecisionMetric()
    assert precision.compute(["a", "b"], ["a", "b", "c"]) == 1.0
    assert precision.compute(["a", "b", "x"], ["a", "b"]) < 1.0
    print("✓ Precision metric works")

    # Test recall
    recall = RecallMetric()
    assert recall.compute(["a", "b"], ["a", "b"]) == 1.0
    assert recall.compute(["a"], ["a", "b"]) == 0.5
    print("✓ Recall metric works")

    # Test F1
    f1 = F1Metric()
    score = f1.compute(["a", "b"], ["a", "b", "c"])
    assert 0 <= score <= 1
    print("✓ F1 metric works")


def test_evaluator():
    """Test that evaluator can be instantiated."""
    print("\nTesting evaluator...")

    evaluator = AgentEvaluator(
        db_path="data/br_funds.db",
        output_dir="evals/results"
    )

    assert len(evaluator.metrics) > 0
    print("✓ Evaluator instantiates correctly")


def test_test_cases():
    """Test that test cases can be loaded."""
    print("\nTesting test case loading...")

    from evals.test_cases import (
        get_all_suites,
        get_suite_by_tool,
        get_all_cases,
        get_edge_cases,
    )

    # Load all suites
    suites = get_all_suites()
    assert len(suites) > 0
    print(f"✓ Loaded {len(suites)} test suites")

    # Load semantic suite
    semantic_suite = get_suite_by_tool(ToolType.SEMANTIC_SEARCH)
    assert semantic_suite is not None
    assert len(semantic_suite.cases) > 0
    print(f"✓ Semantic suite has {len(semantic_suite.cases)} cases")

    # Load all cases
    all_cases = get_all_cases()
    assert len(all_cases) > 0
    print(f"✓ Total test cases: {len(all_cases)}")

    # Load edge cases
    edge_cases = get_edge_cases()
    print(f"✓ Edge cases: {len(edge_cases)}")


def test_eval_suite():
    """Test that eval suites work correctly."""
    print("\nTesting eval suite...")

    cases = [
        EvalCase(
            id="test_001",
            name="Test 1",
            description="Test description",
            tool_type=ToolType.SEMANTIC_SEARCH,
            input_query="test query",
            tags=["basic"],
        ),
        EvalCase(
            id="test_002",
            name="Test 2",
            description="Test description",
            tool_type=ToolType.SEMANTIC_SEARCH,
            input_query="test query 2",
            tags=["edge_case"],
            edge_case=True,
        ),
    ]

    suite = EvalSuite(
        name="Test Suite",
        description="Test suite description",
        cases=cases,
        tool_type=ToolType.SEMANTIC_SEARCH,
    )

    assert len(suite.cases) == 2
    print("✓ Suite created with 2 cases")

    # Test filtering
    edge_suite = suite.filter_edge_cases()
    assert len(edge_suite.cases) == 1
    print("✓ Edge case filtering works")

    tag_suite = suite.filter_by_tags(["basic"])
    assert len(tag_suite.cases) == 1
    print("✓ Tag filtering works")


def test_summary():
    """Test summary generation."""
    print("\nTesting summary generation...")

    results = [
        EvalResult(
            case_id="test_001",
            status=EvalStatus.PASSED,
            execution_time_ms=100.0,
            metrics={"accuracy": 1.0},
            passed_checks=["check1"],
        ),
        EvalResult(
            case_id="test_002",
            status=EvalStatus.FAILED,
            execution_time_ms=150.0,
            metrics={"accuracy": 0.5},
            failed_checks=["check2"],
        ),
    ]

    evaluator = AgentEvaluator()
    summary = evaluator.generate_summary(results)

    assert summary.total_cases == 2
    assert summary.passed == 1
    assert summary.failed == 1
    assert summary.pass_rate == 0.5
    print(f"✓ Summary: {summary.passed}/{summary.total_cases} passed ({summary.pass_rate:.0%})")


def main():
    """Run all tests."""
    print("=" * 80)
    print("EVALUATION FRAMEWORK SELF-TEST")
    print("=" * 80)

    try:
        test_models()
        test_metrics()
        test_evaluator()
        test_test_cases()
        test_eval_suite()
        test_summary()

        print("\n" + "=" * 80)
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        print("\nThe evaluation framework is working correctly!")
        print("\nNext steps:")
        print("  1. Run semantic search evaluation: python -m evals.run_semantic_eval")
        print("  2. Run SQL filter evaluation: python -m evals.run_sql_eval")
        print("  3. Run holdings search evaluation: python -m evals.run_holdings_eval")
        print("  4. Run all evaluations: python -m evals.run_all_evals")
        print("\nOr use the quick start script:")
        print("  python evals/quickstart.py")
        return 0

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
