"""Core evaluation models and types."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class EvalStatus(str, Enum):
    """Evaluation status."""
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"


class ToolType(str, Enum):
    """Types of tools in the agent."""
    SEMANTIC_SEARCH = "semantic_search"
    STRUCTURED_FILTER = "structured_filter"
    HOLDINGS_SEARCH = "holdings_search"
    VISUALIZATION = "visualization"
    FUND_DETAILS = "fund_details"
    AGENT_ROUTING = "agent_routing"


@dataclass
class EvalCase:
    """A single evaluation test case."""
    id: str
    name: str
    description: str
    tool_type: ToolType
    input_query: str
    expected_output: Optional[Dict[str, Any]] = None
    expected_funds: Optional[List[str]] = None  # CNPJs or fund names
    expected_companies: Optional[List[str]] = None  # For holdings search
    min_results: Optional[int] = None
    max_results: Optional[int] = None
    expected_sql_pattern: Optional[str] = None
    expected_visualization_type: Optional[str] = None
    should_fail: bool = False
    edge_case: bool = False
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of a single evaluation."""
    case_id: str
    status: EvalStatus
    actual_output: Optional[Dict[str, Any]] = None
    expected_output: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    passed_checks: List[str] = field(default_factory=list)
    failed_checks: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def passed(self) -> bool:
        """Check if evaluation passed."""
        return self.status == EvalStatus.PASSED

    @property
    def accuracy(self) -> float:
        """Calculate accuracy as ratio of passed checks."""
        total = len(self.passed_checks) + len(self.failed_checks)
        if total == 0:
            return 0.0
        return len(self.passed_checks) / total


@dataclass
class EvalSummary:
    """Summary of evaluation results."""
    total_cases: int
    passed: int
    failed: int
    errors: int
    skipped: int
    total_time_ms: float
    avg_time_ms: float
    accuracy: float
    tool_metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    edge_case_results: Dict[str, float] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total_cases == 0:
            return 0.0
        return self.passed / self.total_cases

    @property
    def error_rate(self) -> float:
        """Calculate error rate."""
        if self.total_cases == 0:
            return 0.0
        return self.errors / self.total_cases


@dataclass
class ToolEvalResult:
    """Evaluation results for a specific tool."""
    tool_type: ToolType
    total_cases: int
    passed: int
    failed: int
    errors: int
    avg_latency_ms: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    edge_case_pass_rate: float
    failed_cases: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)
