import json
from pathlib import Path

import pytest

from ml_infra.config import RunConfig
from ml_infra.evaluation import (
    EvaluationCase,
    EvaluationRunner,
    EvaluationSuite,
    ExactMatchEvaluator,
)
from ml_infra.logging import ResearchLogger
from ml_infra.results_db import ResultsDatabase
from ml_infra.tracker import ExperimentTracker


def test_logger_persists_structured_event(
    tmp_path: Path,
) -> None:
    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    logger = ResearchLogger(
        db,
        name="test_logger",
    )

    event = logger.info(
        "experiment.started",
        "Experiment started.",
        data={
            "model": "toy",
            "seed": 42,
        },
    )

    assert event.level == "INFO"
    assert event.event == (
        "experiment.started"
    )

    logs = db.get_logs()

    assert len(logs) == 1
    assert logs[0]["event"] == (
        "experiment.started"
    )
    assert logs[0]["data"] == {
        "model": "toy",
        "seed": 42,
    }

    db.close()


def test_logger_binds_run_context(
    tmp_path: Path,
) -> None:
    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    tracker = ExperimentTracker(db)

    config = RunConfig(
        project="logging-test",
        run_name="context-test",
        seed=42,
    )

    with tracker.run(config) as run:
        logger = ResearchLogger(
            db,
            name="test_logger",
        )

        run_logger = logger.bind(
            run_id=run.run_id
        )

        # Changed event name to distinguish it from the tracker's auto-log
        event = run_logger.info(
            "custom.event",
            "Custom event.",
        )

        assert event.run_id == run.run_id

        logs = db.get_logs(
            run_id=run.run_id
        )

        # 1 auto-log from tracker, 1 manual log from our test
        assert len(logs) == 2

        assert logs[0]["event"] == "run.started"
        assert logs[0]["logger_name"] == "ml_infra.tracker"

        assert logs[1]["event"] == "custom.event"
        assert logs[1]["logger_name"] == "test_logger"
        assert logs[1]["run_id"] == run.run_id

    db.close()


def test_logger_can_write_jsonl(
    tmp_path: Path,
) -> None:
    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    log_file = (
        tmp_path / "logs" / "events.jsonl"
    )

    logger = ResearchLogger(
        db,
        name="jsonl_test",
        jsonl_path=log_file,
    )

    logger.warning(
        "checkpoint.warning",
        "Checkpoint is unusually large.",
        data={
            "size_bytes": 1234,
        },
    )

    assert log_file.exists()

    lines = log_file.read_text(
        encoding="utf-8"
    ).splitlines()

    assert len(lines) == 1

    record = json.loads(
        lines[0]
    )

    assert record["level"] == "WARNING"
    assert record["event"] == (
        "checkpoint.warning"
    )
    assert record["data"]["size_bytes"] == 1234

    db.close()


def test_logger_respects_log_level(
    tmp_path: Path,
) -> None:
    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    logger = ResearchLogger(
        db,
        name="level_test",
        level="WARNING",
    )

    logger.debug(
        "test.debug",
        "Debug event.",
    )

    logger.info(
        "test.info",
        "Info event.",
    )

    logger.warning(
        "test.warning",
        "Warning event.",
    )

    logs = db.get_logs()

    assert len(logs) == 1
    assert logs[0]["event"] == (
        "test.warning"
    )

    db.close()


def test_tracker_emits_lifecycle_logs(
    tmp_path: Path,
) -> None:
    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    tracker = ExperimentTracker(db)

    config = RunConfig(
        project="logging-test",
        run_name="baseline",
        seed=42,
    )

    with tracker.run(config) as run:
        run_id = run.run_id

    logs = db.get_logs(
        run_id=run_id
    )

    events = [
        log["event"]
        for log in logs
    ]

    assert events == [
        "run.started",
        "run.completed",
    ]

    db.close()

def test_tracker_emits_failure_log(
    tmp_path: Path,
) -> None:
    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    tracker = ExperimentTracker(db)

    config = RunConfig(
        project="logging-test",
        run_name="failure",
        seed=42,
    )

    with pytest.raises(RuntimeError), tracker.run(config) as run:
        run_id = run.run_id

        raise RuntimeError(
            "simulated failure"
        )

    logs = db.get_logs(
        run_id=run_id
    )

    events = [
        log["event"]
        for log in logs
    ]

    assert events == [
        "run.started",
        "run.failed",
    ]

    assert logs[1]["data"][
        "exception_type"
    ] == "RuntimeError"

    db.close()

def test_evaluation_emits_lifecycle_logs(
    tmp_path: Path,
) -> None:
    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    suite = EvaluationSuite(
        name="logging-suite",
        cases=(
            EvaluationCase(
                case_id="1",
                input="a",
                expected="a",
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

    logs = db.get_logs(
        evaluation_id=report.evaluation_id
    )

    events = [
        log["event"]
        for log in logs
    ]

    assert events == [
        "evaluation.started",
        "evaluation.completed",
    ]

    assert (
        logs[0]["data"]["suite_name"]
        == "logging-suite"
    )

    db.close()