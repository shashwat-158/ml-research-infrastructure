import time
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from ml_infra.logging import ResearchLogger
from ml_infra.results_db import ResultsDatabase

from .aggregation import summarize
from .evaluators import Evaluator
from .models import (
    CaseEvaluation,
    EvaluationReport,
    EvaluationSuite,
)


def utc_now() -> str:
    """Return current UTC timestamp."""

    return datetime.now(
        timezone.utc
    ).isoformat()


class EvaluationRunner:
    """
    Execute evaluation suites against a system under test.
    """

    def __init__(
        self,
        db: ResultsDatabase,
        logger: ResearchLogger | None = None,
    ):
        self.db = db
        self.logger = (
            logger
            if logger is not None
            else ResearchLogger(
                db,
                name="ml_infra.evaluation",
            )
        )

    def run(
        self,
        suite: EvaluationSuite,
        subject: Callable[[Any], Any],
        evaluators: list[Evaluator],
        *,
        run_id: str | None = None,
        seed: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationReport:
        """
        Execute a suite.

        The subject is called exactly once per case.
        Its output is passed to every evaluator.
        """

        if not evaluators:
            raise ValueError(
                "At least one evaluator is required."
            )

        evaluator_names = [
            evaluator.name
            for evaluator in evaluators
        ]

        if any(
            not name.strip()
            for name in evaluator_names
        ):
            raise ValueError(
                "Evaluator names cannot be empty."
            )

        if len(evaluator_names) != len(
            set(evaluator_names)
        ):
            raise ValueError(
                "Evaluator names must be unique "
                "within one evaluation."
            )

        suite_hash = suite.hash()

        evaluation_id = uuid4().hex

        config = {
            "suite_name": suite.name,
            "suite_hash": suite_hash,
            "evaluators": evaluator_names,
            "metadata": metadata or {},
        }

        self.db.create_evaluation(
            evaluation_id=evaluation_id,
            run_id=run_id,
            suite_name=suite.name,
            suite_hash=suite_hash,
            evaluator_names=evaluator_names,
            config=config,
            seed=seed,
            started_at=utc_now(),
        )

        evaluation_logger = self.logger.bind(
            run_id=run_id,
            evaluation_id=evaluation_id,
        )

        evaluation_logger.info(
            "evaluation.started",
            "Evaluation started.",
            data={
                "suite_name": suite.name,
                "suite_hash": suite_hash,
                "case_count": len(suite.cases),
                "evaluator_names": evaluator_names,
            },
        )

        results: list[CaseEvaluation] = []

        try:
            for case in suite.cases:

                subject_output = subject(
                    case.input
                )

                for evaluator in evaluators:

                    started = time.perf_counter()

                    decision = evaluator.evaluate(
                        case,
                        subject_output,
                    )

                    duration_ms = (
                        time.perf_counter()
                        - started
                    ) * 1000.0

                    result = CaseEvaluation(
                        case_id=case.case_id,
                        evaluator_name=evaluator.name,
                        passed=decision.passed,
                        score=decision.score,
                        explanation=decision.explanation,
                        details=decision.details,
                        duration_ms=duration_ms,
                    )

                    results.append(result)

                    self.db.log_evaluation_result(
                        evaluation_id=evaluation_id,
                        case_id=case.case_id,
                        evaluator_name=evaluator.name,
                        passed=decision.passed,
                        score=decision.score,
                        explanation=decision.explanation,
                        details=decision.details,
                        duration_ms=duration_ms,
                        created_at=utc_now(),
                    )

            summaries = summarize(
                results
            )

            for summary in summaries:

                for metric_name, value in (
                    summary.metrics().items()
                ):
                    self.db.log_evaluation_metric(
                        evaluation_id=evaluation_id,
                        evaluator_name=(
                            summary.evaluator_name
                        ),
                        name=metric_name,
                        value=value,
                        created_at=utc_now(),
                    )

            self.db.complete_evaluation(
                evaluation_id,
                ended_at=utc_now(),
            )

            evaluation_logger.info(
                "evaluation.completed",
                "Evaluation completed.",
                data={
                    "result_count": len(results),
                },
            )

        except Exception as exc:

            self.db.fail_evaluation(
                evaluation_id,
                ended_at=utc_now(),
            )

            evaluation_logger.error(
                "evaluation.failed",
                "Evaluation failed.",
                data={
                    "exception_type": type(exc).__name__,
                    "exception_message": str(exc),
                },
            )

            raise

        return EvaluationReport(
            evaluation_id=evaluation_id,
            suite_name=suite.name,
            suite_hash=suite_hash,
            results=tuple(results),
            summaries=summaries,
        )