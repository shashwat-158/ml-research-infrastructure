from pathlib import Path

from ml_infra.results_db import ResultsDatabase


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
        dataset_version="dataset-v1",
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
    assert run["dataset_version"] == "dataset-v1"
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