from pathlib import Path

import pytest

from ml_infra.config import RunConfig
from ml_infra.results_db import ResultsDatabase
from ml_infra.tracker import ExperimentTracker


def make_config() -> RunConfig:
    return RunConfig(
        project="test-project",
        run_name="tracker-test",
        seed=42,
        parameters={
            "learning_rate": 0.001,
        },
        tags={
            "stage": "test",
        },
    )


def test_tracker_completes_successful_run(
    tmp_path: Path,
) -> None:
    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    tracker = ExperimentTracker(db)

    config = make_config()

    # Dataset version is intentionally omitted here to test the tracker in isolation
    with tracker.run(
        config,
        git_commit="abc123",
    ) as run:

        run.log_metric(
            "train_loss",
            1.25,
            step=1,
        )

        run.log_metric(
            "train_loss",
            1.10,
            step=2,
        )

        run.log_param(
            "optimizer",
            "AdamW",
        )

    runs = db.list_runs(
        project="test-project"
    )

    assert len(runs) == 1

    stored_run = runs[0]

    assert stored_run["status"] == "completed"
    assert stored_run["seed"] == 42
    assert stored_run["git_commit"] == "abc123"

    metrics = db.get_metrics(
        stored_run["run_id"]
    )

    assert len(metrics) == 2


def test_tracker_marks_failed_run(
    tmp_path: Path,
) -> None:
    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    tracker = ExperimentTracker(db)

    config = make_config()

    with pytest.raises(RuntimeError), tracker.run(config) as run:

        run.log_metric(
            "train_loss",
            3.0,
            step=1,
        )

        raise RuntimeError(
            "simulated training failure"
        )

    runs = db.list_runs(
        project="test-project"
    )

    assert len(runs) == 1
    assert runs[0]["status"] == "failed"


def test_artifact_registration(
    tmp_path: Path,
) -> None:
    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    tracker = ExperimentTracker(db)

    config = make_config()

    artifact = tmp_path / "results.txt"

    artifact.write_text(
        "experiment completed",
        encoding="utf-8",
    )

    with tracker.run(config) as run:
        run.log_artifact(
            artifact,
            artifact_type="text",
        )

    artifacts = db.get_artifacts(
        run.run_id
    )

    assert len(artifacts) == 1

    artifact_record = artifacts[0]

    assert artifact_record["name"] == "results.txt"
    assert artifact_record["artifact_type"] == "text"
    assert artifact_record["size_bytes"] > 0
    assert len(artifact_record["sha256"]) == 64