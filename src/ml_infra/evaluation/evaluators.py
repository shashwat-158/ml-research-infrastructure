from collections.abc import Callable
from dataclasses import dataclass
from numbers import Real
from typing import Any, Protocol

from .models import (
    EvaluationCase,
    EvaluationDecision,
)


class Evaluator(Protocol):
    """
    Protocol implemented by all evaluation graders.
    """

    name: str

    def evaluate(
        self,
        case: EvaluationCase,
        actual: Any,
    ) -> EvaluationDecision:
        ...


def _safe_equal(
    actual: Any,
    expected: Any,
) -> bool:
    """
    Perform equality comparison while tolerating
    array-like boolean results.
    """

    try:
        result = actual == expected
    except Exception: # noqa: BLE001
        return False

    if isinstance(result, bool):
        return result

    try:
        return bool(result.all())
    except AttributeError:
        pass
    except ValueError:
        return False

    try:
        return bool(result)
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class ExactMatchEvaluator:
    """
    Evaluator that checks exact equality.
    """

    name: str = "exact_match"

    def evaluate(
        self,
        case: EvaluationCase,
        actual: Any,
    ) -> EvaluationDecision:

        passed = _safe_equal(
            actual,
            case.expected,
        )

        return EvaluationDecision(
            passed=passed,
            score=1.0 if passed else 0.0,
            explanation=(
                "Output exactly matched expected value."
                if passed
                else "Output did not match expected value."
            ),
            details={
                "comparison": "exact",
            },
        )


@dataclass(frozen=True)
class NumericToleranceEvaluator:
    """
    Evaluator for numerical predictions.

    A prediction passes when:

        absolute_error <= max(
            absolute_tolerance,
            relative_tolerance * abs(expected)
        )
    """

    absolute_tolerance: float = 1e-8
    relative_tolerance: float = 1e-5

    name: str = "numeric_tolerance"

    def __post_init__(self) -> None:
        if self.absolute_tolerance < 0:
            raise ValueError(
                "absolute_tolerance cannot be negative."
            )

        if self.relative_tolerance < 0:
            raise ValueError(
                "relative_tolerance cannot be negative."
            )

    def evaluate(
        self,
        case: EvaluationCase,
        actual: Any,
    ) -> EvaluationDecision:

        if (
            isinstance(actual, bool)
            or not isinstance(actual, Real)
        ):
            return EvaluationDecision(
                passed=False,
                score=0.0,
                explanation=(
                    "Actual output is not numeric."
                ),
                details={
                    "error": "non_numeric_actual",
                },
            )

        if (
            isinstance(case.expected, bool)
            or not isinstance(
                case.expected,
                Real,
            )
        ):
            return EvaluationDecision(
                passed=False,
                score=0.0,
                explanation=(
                    "Expected value is not numeric."
                ),
                details={
                    "error": "non_numeric_expected",
                },
            )

        actual_value = float(actual)
        expected_value = float(
            case.expected
        )

        absolute_error = abs(
            actual_value - expected_value
        )

        tolerance = max(
            self.absolute_tolerance,
            self.relative_tolerance
            * abs(expected_value),
        )

        passed = (
            absolute_error <= tolerance
        )

        return EvaluationDecision(
            passed=passed,
            score=1.0 if passed else 0.0,
            explanation=(
                "Prediction is within tolerance."
                if passed
                else "Prediction exceeds tolerance."
            ),
            details={
                "actual": actual_value,
                "expected": expected_value,
                "absolute_error": absolute_error,
                "tolerance": tolerance,
            },
        )


@dataclass(frozen=True)
class CallableEvaluator:
    """
    Adapter for custom grading logic.

    The callable must return EvaluationDecision.
    """

    function: Callable[
        [EvaluationCase, Any],
        EvaluationDecision,
    ]

    name: str = "callable"

    def evaluate(
        self,
        case: EvaluationCase,
        actual: Any,
    ) -> EvaluationDecision:

        decision = self.function(
            case,
            actual,
        )

        if not isinstance(
            decision,
            EvaluationDecision,
        ):
            raise TypeError(
                "CallableEvaluator function must "
                "return EvaluationDecision."
            )

        return decision