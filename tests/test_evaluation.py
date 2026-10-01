import pytest

from ml_infra.evaluation import (
    CallableEvaluator,
    EvaluationCase,
    EvaluationDecision,
    EvaluationRunner,
    EvaluationSuite,
    ExactMatchEvaluator,
    NumericToleranceEvaluator,
)
from ml_infra.results_db import ResultsDatabase


def test_exact_match_evaluator_passes() -> None:
    case = EvaluationCase(
        case_id="case-1",
        input="hello",
        expected="hello",
    )

    evaluator = ExactMatchEvaluator()

    result = evaluator.evaluate(
        case,
        "hello",
    )

    assert result.passed is True
    assert result.score == 1.0


def test_exact_match_evaluator_fails() -> None:
    case = EvaluationCase(
        case_id="case-1",
        input="hello",
        expected="hello",
    )

    evaluator = ExactMatchEvaluator()

    result = evaluator.evaluate(
        case,
        "goodbye",
    )

    assert result.passed is False
    assert result.score == 0.0


def test_numeric_tolerance_evaluator_passes() -> None:
    case = EvaluationCase(
        case_id="case-1",
        input="prediction",
        expected=1.0,
    )

    evaluator = NumericToleranceEvaluator(
        absolute_tolerance=0.01,
        relative_tolerance=0.0,
    )

    result = evaluator.evaluate(
        case,
        1.005,
    )

    assert result.passed is True
    assert result.details["absolute_error"] == pytest.approx(0.005)


def test_numeric_tolerance_evaluator_fails() -> None:
    case = EvaluationCase(
        case_id="case-1",
        input="prediction",
        expected=1.0,
    )

    evaluator = NumericToleranceEvaluator(
        absolute_tolerance=0.01,
        relative_tolerance=0.0,
    )

    result = evaluator.evaluate(
        case,
        1.02,
    )

    assert result.passed is False


def test_callable_evaluator_uses_custom_logic() -> None:
    case = EvaluationCase(
        case_id="case-1",
        input="hello",
        expected="HELLO",
    )

    evaluator = CallableEvaluator(
        name="case_insensitive",
        function=lambda case, actual: (
            EvaluationDecision(
                passed=(
                    str(actual).lower()
                    == str(case.expected).lower()
                ),
                score=1.0
                if (
                    str(actual).lower()
                    == str(case.expected).lower()
                )
                else 0.0,
            )
        ),
    )

    result = evaluator.evaluate(
        case,
        "hello",
    )

    assert result.passed is True
    assert result.score == 1.0


def test_suite_rejects_duplicate_case_ids() -> None:
    with pytest.raises(ValueError):
        EvaluationSuite(
            name="duplicate-test",
            cases=(
                EvaluationCase(
                    case_id="same",
                    input=1,
                ),
                EvaluationCase(
                    case_id="same",
                    input=2,
                ),
            ),
        )


def test_suite_hash_is_deterministic() -> None:
    suite_a = EvaluationSuite(
        name="test",
        cases=(
            EvaluationCase(
                case_id="1",
                input="a",
                expected="b",
            ),
            EvaluationCase(
                case_id="2",
                input="c",
                expected="d",
            ),
        ),
    )

    suite_b = EvaluationSuite(
        name="test",
        cases=(
            EvaluationCase(
                case_id="1",
                input="a",
                expected="b",
            ),
            EvaluationCase(
                case_id="2",
                input="c",
                expected="d",
            ),
        ),
    )

    assert suite_a.hash() == suite_b.hash()


def test_suite_hash_changes_when_expected_changes() -> None:
    suite_a = EvaluationSuite(
        name="test",
        cases=(
            EvaluationCase(
                case_id="1",
                input="a",
                expected="b",
            ),
        ),
    )

    suite_b = EvaluationSuite(
        name="test",
        cases=(
            EvaluationCase(
                case_id="1",
                input="a",
                expected="c",
            ),
        ),
    )

    assert suite_a.hash() != suite_b.hash()

def test_runner_evaluates_subject_once_per_case(
    tmp_path,
) -> None:

    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    suite = EvaluationSuite(
        name="identity",
        cases=(
            EvaluationCase(
                case_id="1",
                input="a",
                expected="a",
            ),
            EvaluationCase(
                case_id="2",
                input="b",
                expected="b",
            ),
        ),
    )

    calls = []

    def subject(value):
        calls.append(value)
        return value

    runner = EvaluationRunner(db)

    report = runner.run(
        suite,
        subject,
        [
            ExactMatchEvaluator(),
            ExactMatchEvaluator(
                name="second_exact_match"
            ),
        ],
    )

    assert calls == [
        "a",
        "b",
    ]

    assert len(report.results) == 4

    assert len(report.summaries) == 2

    for summary in report.summaries:
        assert summary.total_cases == 2
        assert summary.passed_cases == 2
        assert summary.failed_cases == 0
        assert summary.pass_rate == 1.0
        assert summary.mean_score == 1.0

    db.close()

def test_runner_persists_evaluation(
    tmp_path,
) -> None:

    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    suite = EvaluationSuite(
        name="identity",
        cases=(
            EvaluationCase(
                case_id="1",
                input="a",
                expected="a",
            ),
            EvaluationCase(
                case_id="2",
                input="b",
                expected="x",
            ),
        ),
    )

    runner = EvaluationRunner(db)

    report = runner.run(
        suite,
        lambda value: value,
        [
            ExactMatchEvaluator()
        ],
    )

    evaluation = db.get_evaluation(
        report.evaluation_id
    )

    assert evaluation is not None
    assert evaluation["suite_name"] == (
        "identity"
    )
    assert evaluation["suite_hash"] == (
        suite.hash()
    )
    assert evaluation["status"] == (
        "completed"
    )
    assert evaluation["evaluator_names"] == [
        "exact_match"
    ]

    results = db.get_evaluation_results(
        report.evaluation_id
    )

    assert len(results) == 2

    assert results[0]["case_id"] == "1"
    assert results[0]["passed"] is True

    assert results[1]["case_id"] == "2"
    assert results[1]["passed"] is False

    metrics = db.get_evaluation_metrics(
        report.evaluation_id
    )

    metric_map = {
        (
            metric["evaluator_name"],
            metric["name"],
        ): metric["value"]
        for metric in metrics
    }

    assert metric_map[
        ("exact_match", "cases_total")
    ] == 2.0

    assert metric_map[
        ("exact_match", "cases_passed")
    ] == 1.0

    assert metric_map[
        ("exact_match", "cases_failed")
    ] == 1.0

    assert metric_map[
        ("exact_match", "pass_rate")
    ] == 0.5

    db.close()

def test_failed_subject_marks_evaluation_failed(
    tmp_path,
) -> None:

    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    suite = EvaluationSuite(
        name="failure-test",
        cases=(
            EvaluationCase(
                case_id="1",
                input="boom",
            ),
        ),
    )

    def failing_subject(value):
        raise RuntimeError(
            "simulated model failure"
        )

    runner = EvaluationRunner(db)

    with pytest.raises(RuntimeError):
        runner.run(
            suite,
            failing_subject,
            [
                ExactMatchEvaluator()
            ],
        )

    evaluations = db._connection.execute(
        """
        SELECT evaluation_id, status
        FROM evaluation_runs
        """
    ).fetchall()

    assert len(evaluations) == 1

    assert evaluations[0]["status"] == (
        "failed"
    )

    db.close()

def test_evaluation_can_attach_to_run(
    tmp_path,
) -> None:

    from ml_infra.config import RunConfig
    from ml_infra.tracker import ExperimentTracker

    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    tracker = ExperimentTracker(db)

    config = RunConfig(
        project="evaluation-test",
        run_name="model-run",
        seed=42,
    )

    with tracker.run(config) as run:

        suite = EvaluationSuite(
            name="identity",
            cases=(
                EvaluationCase(
                    case_id="1",
                    input="hello",
                    expected="hello",
                ),
            ),
        )

        runner = EvaluationRunner(db)

        report = runner.run(
            suite,
            lambda value: value,
            [
                ExactMatchEvaluator()
            ],
            run_id=run.run_id,
            seed=42,
        )

    evaluation = db.get_evaluation(
        report.evaluation_id
    )

    assert evaluation is not None
    assert evaluation["run_id"] == (
        run.run_id
    )
    assert evaluation["seed"] == 42
    assert evaluation["status"] == "completed"

    db.close()

'''Run
 │
 │ run_id
 ▼
Evaluation
 │
 ├── suite_hash
 ├── evaluators
 │
 ├── case results
 │
 └── aggregate metrics'''