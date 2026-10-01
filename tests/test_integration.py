from pathlib import Path

from ml_infra.checkpoints import CheckpointManager
from ml_infra.config import RunConfig
from ml_infra.results_db import ResultsDatabase
from ml_infra.tracker import ExperimentTracker


def test_end_to_end_experiment_workflow(
    tmp_path: Path,
) -> None:

    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    config = RunConfig(
        project="integration-test",
        run_name="baseline",
        seed=42,
        parameters={
            "learning_rate": 0.001,
            "batch_size": 32,
        },
        tags={
            "stage": "integration",
        },
    )

    tracker = ExperimentTracker(db)

    checkpoint_manager = CheckpointManager(
        db,
        root_dir=tmp_path / "runs",
    )

    artifact = tmp_path / "metrics.txt"

    artifact.write_text(
        "final validation loss = 2.1",
        encoding="utf-8",
    )

    with tracker.run(
        config,
        git_commit="integration-test-commit",
        dataset_version="dataset-v1",
    ) as run:

        run.log_metric(
            "val_loss",
            2.5,
            step=100,
        )

        checkpoint_manager.save_bytes(
                run.run_id,
                "model.pt",
                b"checkpoint-step-100",
                step=100,
                metric_name="val_loss",
                metric_value=2.5,
                mode="min",
        )

        run.log_metric(
            "val_loss",
            2.1,
            step=200,
        )

        second_checkpoint = (
            checkpoint_manager.save_bytes(
                run.run_id,
                "model.pt",
                b"checkpoint-step-200",
                step=200,
                metric_name="val_loss",
                metric_value=2.1,
                mode="min",
            )
        )

        run.log_artifact(
            artifact,
            artifact_type="text",
        )

        run_id = run.run_id

    # ---------------------------------------------------------
    # Run metadata
    # ---------------------------------------------------------

    stored_run = db.get_run(run_id)

    assert stored_run is not None
    assert stored_run["project"] == "integration-test"
    assert stored_run["run_name"] == "baseline"
    assert stored_run["seed"] == 42
    assert stored_run["config_hash"]
    assert stored_run["git_commit"] == (
        "integration-test-commit"
    )
    assert stored_run["dataset_version"] == (
        "dataset-v1"
    )
    assert stored_run["status"] == "completed"

    # ---------------------------------------------------------
    # Metrics
    # ---------------------------------------------------------

    metrics = db.get_metrics(
        run_id,
        "val_loss",
    )

    assert len(metrics) == 2
    assert metrics[0]["value"] == 2.5
    assert metrics[1]["value"] == 2.1

    # ---------------------------------------------------------
    # Latest checkpoint
    # ---------------------------------------------------------

    latest = checkpoint_manager.latest(
        run_id
    )

    assert latest is not None
    assert latest.checkpoint_id == (
        second_checkpoint.checkpoint_id
    )

    # ---------------------------------------------------------
    # Best checkpoint
    # ---------------------------------------------------------

    best = checkpoint_manager.best(
        run_id,
        "val_loss",
    )

    assert best is not None
    assert best.checkpoint_id == (
        second_checkpoint.checkpoint_id
    )

    # ---------------------------------------------------------
    # Exactly one best checkpoint
    # ---------------------------------------------------------

    checkpoints = checkpoint_manager.list(
        run_id
    )

    best_checkpoints = [
        checkpoint
        for checkpoint in checkpoints
        if checkpoint.metric_name == "val_loss"
        and checkpoint.is_best
    ]

    assert len(best_checkpoints) == 1

    # ---------------------------------------------------------
    # Checksum-verified loading
    # ---------------------------------------------------------

    restored = checkpoint_manager.load(
        second_checkpoint.checkpoint_id,
        lambda path: path.read_bytes(),
    )

    assert restored == b"checkpoint-step-200"

    # ---------------------------------------------------------
    # Artifact metadata
    # ---------------------------------------------------------

    artifacts = db.get_artifacts(
        run_id
    )

    assert len(artifacts) == 1
    assert artifacts[0]["name"] == "metrics.txt"
    assert artifacts[0]["artifact_type"] == "text"

    db.close()