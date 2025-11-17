"""Evaluation metrics for agent tools."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import time


class Metric(ABC):
    """Base class for evaluation metrics."""

    @abstractmethod
    def compute(
        self,
        actual: Any,
        expected: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """Compute the metric value."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Get metric name."""
        pass


class AccuracyMetric(Metric):
    """Accuracy metric - ratio of correct predictions."""

    @property
    def name(self) -> str:
        return "accuracy"

    def compute(
        self,
        actual: Any,
        expected: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute accuracy for different types of outputs.

        For fund results: checks if expected funds are in actual results
        For classifications: checks if actual matches expected
        """
        if expected is None:
            return 1.0

        # For list comparisons (e.g., fund CNPJs)
        if isinstance(expected, list) and isinstance(actual, list):
            if len(expected) == 0:
                return 1.0 if len(actual) >= 0 else 0.0

            matches = sum(1 for item in expected if item in actual)
            return matches / len(expected)

        # For exact matches
        if actual == expected:
            return 1.0

        # For fuzzy string matching
        if isinstance(actual, str) and isinstance(expected, str):
            actual_lower = actual.lower().strip()
            expected_lower = expected.lower().strip()
            if actual_lower == expected_lower:
                return 1.0
            if expected_lower in actual_lower or actual_lower in expected_lower:
                return 0.8
            return 0.0

        return 0.0


class PrecisionMetric(Metric):
    """Precision metric - ratio of true positives to all positive predictions."""

    @property
    def name(self) -> str:
        return "precision"

    def compute(
        self,
        actual: Any,
        expected: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute precision: TP / (TP + FP)

        For fund search: ratio of relevant funds in results
        """
        if not isinstance(actual, list) or not isinstance(expected, list):
            return 1.0 if actual == expected else 0.0

        if len(actual) == 0:
            return 0.0

        true_positives = sum(1 for item in actual if item in expected)
        return true_positives / len(actual)


class RecallMetric(Metric):
    """Recall metric - ratio of true positives to all actual positives."""

    @property
    def name(self) -> str:
        return "recall"

    def compute(
        self,
        actual: Any,
        expected: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute recall: TP / (TP + FN)

        For fund search: ratio of expected funds that were found
        """
        if not isinstance(actual, list) or not isinstance(expected, list):
            return 1.0 if actual == expected else 0.0

        if len(expected) == 0:
            return 1.0

        true_positives = sum(1 for item in expected if item in actual)
        return true_positives / len(expected)


class F1Metric(Metric):
    """F1 Score - harmonic mean of precision and recall."""

    def __init__(self):
        self.precision = PrecisionMetric()
        self.recall = RecallMetric()

    @property
    def name(self) -> str:
        return "f1_score"

    def compute(
        self,
        actual: Any,
        expected: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """Compute F1 score: 2 * (precision * recall) / (precision + recall)"""
        p = self.precision.compute(actual, expected, metadata)
        r = self.recall.compute(actual, expected, metadata)

        if p + r == 0:
            return 0.0

        return 2 * (p * r) / (p + r)


class LatencyMetric(Metric):
    """Latency metric - execution time measurement."""

    @property
    def name(self) -> str:
        return "latency_ms"

    def compute(
        self,
        actual: Any,
        expected: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Get latency from metadata or measure directly.

        Returns latency in milliseconds.
        """
        if metadata and "execution_time_ms" in metadata:
            return metadata["execution_time_ms"]
        return 0.0


class ErrorRateMetric(Metric):
    """Error rate metric - ratio of errors to total attempts."""

    @property
    def name(self) -> str:
        return "error_rate"

    def compute(
        self,
        actual: Any,
        expected: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute error rate based on success flag.

        Returns 1.0 if error, 0.0 if success.
        """
        if metadata and "success" in metadata:
            return 0.0 if metadata["success"] else 1.0

        # If actual is None or empty when expected is not, consider it an error
        if expected is not None and actual is None:
            return 1.0

        return 0.0


class RelevanceMetric(Metric):
    """Relevance metric - how relevant are the results to the query."""

    @property
    def name(self) -> str:
        return "relevance"

    def compute(
        self,
        actual: Any,
        expected: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute relevance score based on result characteristics.

        For semantic search: uses similarity scores
        For other tools: uses heuristics
        """
        if metadata and "relevance_scores" in metadata:
            scores = metadata["relevance_scores"]
            return sum(scores) / len(scores) if scores else 0.0

        # Default to accuracy if no specific relevance data
        return AccuracyMetric().compute(actual, expected, metadata)


class CompletenessMetric(Metric):
    """Completeness metric - how complete are the results."""

    @property
    def name(self) -> str:
        return "completeness"

    def compute(
        self,
        actual: Any,
        expected: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute completeness based on result coverage.

        Checks if all required fields are present.
        """
        if metadata and "required_fields" in metadata:
            required = metadata["required_fields"]
            if isinstance(actual, dict):
                present = sum(1 for field in required if field in actual and actual[field] is not None)
                return present / len(required) if required else 1.0

        # Default check: non-empty results
        if actual is None:
            return 0.0
        if isinstance(actual, (list, dict, str)):
            return 1.0 if len(actual) > 0 else 0.0
        return 1.0


class RobustnessMetric(Metric):
    """Robustness metric - how well the tool handles edge cases."""

    @property
    def name(self) -> str:
        return "robustness"

    def compute(
        self,
        actual: Any,
        expected: Any,
        metadata: Optional[Dict[str, Any]] = None
    ) -> float:
        """
        Compute robustness score.

        1.0 if tool handled edge case gracefully
        0.5 if tool returned partial results
        0.0 if tool failed
        """
        if metadata and "success" in metadata:
            if not metadata["success"]:
                # Check if error was handled gracefully
                if metadata.get("error_handled", False):
                    return 0.5
                return 0.0
            return 1.0

        return 1.0 if actual is not None else 0.0
