import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any


def _canonical_json(value: Any) -> str:
    """
    Serialize JSON-compatible values deterministically.
    """

    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )

    except (TypeError, ValueError) as exc:
        raise TypeError(
            "Evaluation cases, metadata, and details "
            "must be JSON serializable."
        ) from exc


@dataclass(frozen=True)
class EvaluationCase:
    """
    One atomic evaluation task.
    """

    case_id: str

    input: Any
    expected: Any = None

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.case_id.strip():
            raise ValueError(
                "case_id cannot be empty."
            )


@dataclass(frozen=True)
class EvaluationSuite:
    """
    Immutable collection of evaluation cases.
    """

    name: str

    cases: tuple[EvaluationCase, ...]

    metadata: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError(
                "Evaluation suite name cannot be empty."
            )

        if not self.cases:
            raise ValueError(
                "Evaluation suite must contain "
                "at least one case."
            )

        case_ids = [
            case.case_id
            for case in self.cases
        ]

        if len(case_ids) != len(set(case_ids)):
            raise ValueError(
                "Evaluation suite contains duplicate "
                "case IDs."
            )

    def hash(self) -> str:
        """
        Compute deterministic identity for the suite.
        """

        payload = {
            "schema_version": 1,
            "name": self.name,
            "metadata": self.metadata,
            "cases": [
                {
                    "case_id": case.case_id,
                    "input": case.input,
                    "expected": case.expected,
                    "metadata": case.metadata,
                }
                for case in self.cases
            ],
        }

        canonical = _canonical_json(
            payload
        )

        return hashlib.sha256(
            canonical.encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True)
class EvaluationDecision:
    """
    Decision returned by an evaluator.
    """

    passed: bool | None

    score: float | None

    explanation: str | None = None

    details: dict[str, Any] = field(
        default_factory=dict
    )

    def __post_init__(self) -> None:
        if self.score is not None:
            if not math.isfinite(
                float(self.score)
            ):
                raise ValueError(
                    "score must be finite."
                )


@dataclass(frozen=True)
class CaseEvaluation:
    """
    Result from evaluating one case with one evaluator.
    """

    case_id: str
    evaluator_name: str

    passed: bool | None
    score: float | None

    explanation: str | None

    details: dict[str, Any]

    duration_ms: float


@dataclass(frozen=True)
class EvaluatorSummary:
    """
    Aggregate result for one evaluator across a suite.
    """

    evaluator_name: str

    total_cases: int
    passed_cases: int
    failed_cases: int
    ungraded_cases: int

    pass_rate: float | None
    mean_score: float | None

    def metrics(self) -> dict[str, float]:
        """
        Convert summary into queryable metrics.
        """

        metrics = {
            "cases_total": float(
                self.total_cases
            ),
            "cases_passed": float(
                self.passed_cases
            ),
            "cases_failed": float(
                self.failed_cases
            ),
            "cases_ungraded": float(
                self.ungraded_cases
            ),
        }

        if self.pass_rate is not None:
            metrics["pass_rate"] = (
                self.pass_rate
            )

        if self.mean_score is not None:
            metrics["mean_score"] = (
                self.mean_score
            )

        return metrics


@dataclass(frozen=True)
class EvaluationReport:
    """
    Complete report for one evaluation execution.
    """

    evaluation_id: str

    suite_name: str
    suite_hash: str

    results: tuple[CaseEvaluation, ...]

    summaries: tuple[EvaluatorSummary, ...]