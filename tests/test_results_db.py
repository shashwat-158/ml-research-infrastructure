import sqlite3
from pathlib import Path

import pytest

from ml_infra.config import RunConfig
from ml_infra.results_db import ResultsDatabase
from ml_infra.tracker import ExperimentTracker


def test_create_and_retrieve_run(tmp_path: Path) -> None:
    db_path = tmp_path / "results.sqlite"

    db = ResultsDatabase(db_path)

    db.create_run(
        run_id="run-001",
        project="test-project",
        run_name="baseline",
        seed=42,
        config_hash="abc123",
        config={
            "project": "test-project",
            "run_name": "baseline",
            "seed": 42,
        },
        git_commit="deadbeef",
        dataset_version=None,
        started_at="2026-10-01T00:00:00+00:00",
        tags={
            "stage": "baseline",
        },
    )

    run = db.get_run("run-001")

    assert run is not None
    assert run["project"] == "test-project"
    assert run["run_name"] == "baseline"
    assert run["seed"] == 42
    assert run["config_hash"] == "abc123"
    assert run["git_commit"] == "deadbeef"
    assert run["dataset_version"] is None
    assert run["status"] == "running"
    assert run["config"]["seed"] == 42

    db.close()


def test_metrics_are_stored_and_queryable(tmp_path: Path) -> None:
    db = ResultsDatabase(tmp_path / "results.sqlite")

    db.create_run(
        run_id="run-001",
        project="test-project",
        run_name="baseline",
        seed=42,
        config_hash="abc123",
        config={"seed": 42},
        git_commit=None,
        dataset_version=None,
        started_at="2026-10-01T00:00:00+00:00",
        tags={},
    )

    db.log_metric(
        run_id="run-001",
        name="train_loss",
        value=2.5,
        step=1,
        timestamp="2026-10-01T00:01:00+00:00",
    )

    db.log_metric(
        run_id="run-001",
        name="train_loss",
        value=2.1,
        step=2,
        timestamp="2026-10-01T00:02:00+00:00",
    )

    metrics = db.get_metrics(
        "run-001",
        name="train_loss",
    )

    assert len(metrics) == 2
    assert metrics[0]["value"] == 2.5
    assert metrics[1]["value"] == 2.1
    assert metrics[0]["step"] == 1
    assert metrics[1]["step"] == 2

    db.close()


def test_run_can_complete(tmp_path: Path) -> None:
    db = ResultsDatabase(tmp_path / "results.sqlite")

    db.create_run(
        run_id="run-001",
        project="test-project",
        run_name="baseline",
        seed=42,
        config_hash="abc123",
        config={"seed": 42},
        git_commit=None,
        dataset_version=None,
        started_at="2026-10-01T00:00:00+00:00",
        tags={},
    )

    db.complete_run(
        "run-001",
        "2026-10-01T00:03:00+00:00",
    )

    run = db.get_run("run-001")

    assert run is not None
    assert run["status"] == "completed"
    assert run["ended_at"] == "2026-10-01T00:03:00+00:00"

    db.close()


def test_run_can_fail(tmp_path: Path) -> None:
    db = ResultsDatabase(tmp_path / "results.sqlite")

    db.create_run(
        run_id="run-001",
        project="test-project",
        run_name="baseline",
        seed=42,
        config_hash="abc123",
        config={"seed": 42},
        git_commit=None,
        dataset_version=None,
        started_at="2026-10-01T00:00:00+00:00",
        tags={},
    )

    db.fail_run(
        "run-001",
        "2026-10-01T00:03:00+00:00",
    )

    run = db.get_run("run-001")

    assert run is not None
    assert run["status"] == "failed"

    db.close()

def test_existing_v1_database_is_migrated(
    tmp_path,
) -> None:
    db_path = tmp_path / "legacy.sqlite"

    connection = sqlite3.connect(db_path)

    connection.execute(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY
        )
        """
    )

    connection.commit()
    connection.close()

    db = ResultsDatabase(db_path)

    assert db.schema_version() == 5

    tables = db._connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name = 'checkpoints'
        """
    ).fetchone()

    assert tables is not None

    dataset_tables = db._connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name IN (
            'datasets',
            'dataset_versions',
            'dataset_files'
        )
        ORDER BY name
        """
    ).fetchall()

    assert [
        row["name"]
        for row in dataset_tables
    ] == [
        "dataset_files",
        "dataset_versions",
        "datasets",
    ]

    evaluation_tables = db._connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name IN (
                'evaluation_runs',
                'evaluation_results',
                'evaluation_metrics'
            )
            ORDER BY name
            """
    ).fetchall()
    
    assert [
            row["name"]
            for row in evaluation_tables
        ] == [
            "evaluation_metrics",
            "evaluation_results",
            "evaluation_runs",
        ]

    log_tables = db._connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type = 'table'
            AND name = 'logs'
            """
        ).fetchall()
    
    assert [
            row["name"]
            for row in log_tables
        ] == [
            "logs"
        ]

    db.close()


def test_new_database_starts_at_current_version(
    tmp_path,
) -> None:
    db = ResultsDatabase(
        tmp_path / "new.sqlite"
    )

    assert db.schema_version() == 5

    tables = db._connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
          AND name = 'checkpoints'
        """
    ).fetchone()

    assert tables is not None

    dataset_tables = db._connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name IN (
            'datasets',
            'dataset_versions',
            'dataset_files'
        )
        ORDER BY name
        """
    ).fetchall()

    assert [
        row["name"]
        for row in dataset_tables
    ] == [
        "dataset_files",
        "dataset_versions",
        "datasets",
    ]

    evaluation_tables = db._connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name IN (
            'evaluation_runs',
            'evaluation_results',
            'evaluation_metrics'
        )
        ORDER BY name
        """
    ).fetchall()

    assert [
        row["name"]
        for row in evaluation_tables
    ] == [
        "evaluation_metrics",
        "evaluation_results",
        "evaluation_runs",
    ]

    log_tables = db._connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type = 'table'
        AND name = 'logs'
        """
    ).fetchall()

    assert [
        row["name"]
        for row in log_tables
    ] == [
        "logs"
    ]

    db.close()

def test_run_requires_registered_dataset_version(
    tmp_path,
) -> None:

    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    tracker = ExperimentTracker(db)

    config = RunConfig(
        project="dataset-test",
        run_name="baseline",
    )

    with pytest.raises(ValueError), tracker.run(
        config,
        dataset_version="does-not-exist",
    ):
        pass

    db.close()

def test_log_event_is_stored_and_queryable(
    tmp_path,
) -> None:
    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    db.log_event(
        timestamp="2026-10-01T10:00:00+00:00",
        level="INFO",
        logger_name="test",
        event="test.started",
        message="Test started.",
        run_id=None,
        evaluation_id=None,
        data={
            "attempt": 1,
            "mode": "unit",
        },
    )

    logs = db.get_logs()

    assert len(logs) == 1

    assert logs[0]["level"] == "INFO"
    assert logs[0]["logger_name"] == "test"
    assert logs[0]["event"] == "test.started"
    assert logs[0]["message"] == (
        "Test started."
    )

    assert logs[0]["data"] == {
        "attempt": 1,
        "mode": "unit",
    }

    db.close()

def test_logs_can_be_filtered(
    tmp_path,
) -> None:
    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    db.create_run(
        run_id="run-1",
        project="test-project",
        run_name="run-1",
        seed=42,
        config_hash="hash-1",
        config={
            "seed": 42,
        },
        git_commit=None,
        dataset_version=None,
        started_at="2026-10-01T10:00:00+00:00",
        tags={},
    )

    db.create_run(
        run_id="run-2",
        project="test-project",
        run_name="run-2",
        seed=43,
        config_hash="hash-2",
        config={
            "seed": 43,
        },
        git_commit=None,
        dataset_version=None,
        started_at="2026-10-01T10:00:01+00:00",
        tags={},
    )

    db.log_event(
        timestamp="2026-10-01T10:00:00+00:00",
        level="INFO",
        logger_name="test",
        event="run.started",
        message="Run started.",
        run_id="run-1",
        evaluation_id=None,
        data={},
    )

    db.log_event(
        timestamp="2026-10-01T10:01:00+00:00",
        level="ERROR",
        logger_name="test",
        event="run.failed",
        message="Run failed.",
        run_id="run-1",
        evaluation_id=None,
        data={
            "reason": "oom",
        },
    )

    db.log_event(
        timestamp="2026-10-01T10:02:00+00:00",
        level="INFO",
        logger_name="test",
        event="run.started",
        message="Another run started.",
        run_id="run-2",
        evaluation_id=None,
        data={},
    )

    run_one_logs = db.get_logs(
        run_id="run-1"
    )

    assert len(run_one_logs) == 2

    error_logs = db.get_logs(
        level="ERROR"
    )

    assert len(error_logs) == 1
    assert error_logs[0]["event"] == (
        "run.failed"
    )

    started_logs = db.get_logs(
        event="run.started"
    )

    assert len(started_logs) == 2

    db.close()