"""Evaluation framework for the Financial Agent and its tools."""

from .evaluator import AgentEvaluator, EvalResult, EvalSuite
from .metrics import (
    AccuracyMetric,
    PrecisionMetric,
    RecallMetric,
    F1Metric,
    LatencyMetric,
    ErrorRateMetric,
)

__all__ = [
    "AgentEvaluator",
    "EvalResult",
    "EvalSuite",
    "AccuracyMetric",
    "PrecisionMetric",
    "RecallMetric",
    "F1Metric",
    "LatencyMetric",
    "ErrorRateMetric",
]
