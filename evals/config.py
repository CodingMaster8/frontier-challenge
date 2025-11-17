"""Configuration for evaluation framework."""

from dataclasses import dataclass
from typing import Optional


@dataclass
class EvalConfig:
    """Configuration for evaluations."""

    # Database
    db_path: str = "data/br_funds.db"

    # Output
    output_dir: str = "evals/results"
    export_json: bool = True
    export_html: bool = True

    # Execution
    parallel_execution: bool = False
    max_workers: int = 4
    timeout_seconds: float = 30.0

    # Thresholds
    pass_rate_threshold: float = 0.75
    error_rate_threshold: float = 0.15
    accuracy_threshold: float = 0.70

    # Tool-specific thresholds
    semantic_pass_rate: float = 0.80
    sql_pass_rate: float = 0.75
    holdings_pass_rate: float = 0.80

    # Test filtering
    run_edge_cases: bool = True
    run_basic_cases: bool = True
    tags_to_include: Optional[list] = None
    tags_to_exclude: Optional[list] = None

    # Reporting
    verbose: bool = True
    show_passed_checks: bool = False
    show_failed_checks: bool = True
    max_failed_to_show: int = 10

    # Performance
    measure_memory: bool = False
    profile_execution: bool = False


# Default configuration
DEFAULT_CONFIG = EvalConfig()

# Quick test configuration (subset of cases)
QUICK_TEST_CONFIG = EvalConfig(
    run_edge_cases=False,
    tags_to_include=["basic"],
    verbose=False,
)

# CI/CD configuration
CI_CONFIG = EvalConfig(
    parallel_execution=True,
    timeout_seconds=60.0,
    export_json=True,
    export_html=True,
    verbose=True,
)

# Comprehensive configuration (all tests, all features)
COMPREHENSIVE_CONFIG = EvalConfig(
    run_edge_cases=True,
    run_basic_cases=True,
    parallel_execution=False,  # Sequential for better debugging
    measure_memory=True,
    profile_execution=True,
    verbose=True,
    show_passed_checks=True,
    show_failed_checks=True,
)
