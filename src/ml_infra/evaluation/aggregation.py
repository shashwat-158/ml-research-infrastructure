from collections import defaultdict

from .models import (
    CaseEvaluation,
    EvaluatorSummary,
)


def summarize(
    results: list[CaseEvaluation],
) -> tuple[EvaluatorSummary, ...]:
    """
    Aggregate case-level results by evaluator.
    """

    grouped: dict[
        str,
        list[CaseEvaluation],
    ] = defaultdict(list)

    for result in results:
        grouped[
            result.evaluator_name
        ].append(result)

    summaries = []

    for evaluator_name in sorted(
        grouped
    ):
        evaluator_results = grouped[
            evaluator_name
        ]

        total_cases = len(
            evaluator_results
        )

        passed_cases = sum(
            result.passed is True
            for result in evaluator_results
        )

        failed_cases = sum(
            result.passed is False
            for result in evaluator_results
        )

        ungraded_cases = sum(
            result.passed is None
            for result in evaluator_results
        )

        graded_cases = (
            passed_cases
            + failed_cases
        )

        pass_rate = (
            passed_cases / graded_cases
            if graded_cases > 0
            else None
        )

        scored_results = [
            result.score
            for result in evaluator_results
            if result.score is not None
        ]

        mean_score = (
            sum(scored_results)
            / len(scored_results)
            if scored_results
            else None
        )

        summaries.append(
            EvaluatorSummary(
                evaluator_name=evaluator_name,
                total_cases=total_cases,
                passed_cases=passed_cases,
                failed_cases=failed_cases,
                ungraded_cases=ungraded_cases,
                pass_rate=pass_rate,
                mean_score=mean_score,
            )
        )

    return tuple(summaries)