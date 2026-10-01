from contextlib import contextmanager

import pytest

from ml_infra.checkpoints import CheckpointManager
from ml_infra.checkpoints.manager import (
    CheckpointCorruptionError,
)
from ml_infra.config import RunConfig
from ml_infra.results_db import ResultsDatabase
from ml_infra.tracker import ExperimentTracker


@contextmanager
def running_run(db: ResultsDatabase):
    tracker = ExperimentTracker(db)

    config = RunConfig(
        project="checkpoint-test",
        run_name="baseline",
        seed=42,
    )

    with tracker.run(config) as run:
        yield run


def test_save_and_load_checkpoint(tmp_path):
    db = ResultsDatabase(tmp_path / "results.sqlite")
    manager = CheckpointManager(
        db,
        root_dir=tmp_path / "runs",
    )

    with running_run(db) as run:
        checkpoint = manager.save_bytes(
            run.run_id,
            "model.pt",
            b"model state",
            step=100,
        )

        assert checkpoint.run_id == run.run_id
        assert checkpoint.step == 100
        assert checkpoint.size_bytes == len(b"model state")
        assert checkpoint.path.exists()

        restored = manager.load(
            checkpoint.checkpoint_id,
            lambda path: path.read_bytes(),
        )

        assert restored == b"model state"

    db.close()


def test_latest_checkpoint_is_updated(tmp_path):
    db = ResultsDatabase(tmp_path / "results.sqlite")
    manager = CheckpointManager(
        db,
        root_dir=tmp_path / "runs",
    )

    with running_run(db) as run:
        manager.save_bytes(
            run.run_id,
            "model.pt",
            b"step100",
            step=100,
        )

        second = manager.save_bytes(
            run.run_id,
            "model.pt",
            b"step200",
            step=200,
        )

        # exactly one latest checkpoint exists
        latest = manager.latest(run.run_id)

        assert latest is not None
        assert latest.checkpoint_id == second.checkpoint_id
        assert latest.is_latest

        checkpoints = manager.list(run.run_id)

        latest_checkpoints = [
            checkpoint
            for checkpoint in checkpoints
            if checkpoint.is_latest
        ]

        assert len(latest_checkpoints) == 1
        assert latest_checkpoints[0].checkpoint_id == second.checkpoint_id

    db.close()


def test_best_checkpoint_with_min_mode(tmp_path):
    db = ResultsDatabase(tmp_path / "results.sqlite")
    manager = CheckpointManager(
        db,
        root_dir=tmp_path / "runs",
    )

    with running_run(db) as run:
        manager.save_bytes(
            run.run_id,
            "model.pt",
            b"first",
            step=100,
            metric_name="val_loss",
            metric_value=2.5,
            mode="min",
        )

        second = manager.save_bytes(
            run.run_id,
            "model.pt",
            b"second",
            step=200,
            metric_name="val_loss",
            metric_value=2.1,
            mode="min",
        )

        manager.save_bytes(
            run.run_id,
            "model.pt",
            b"third",
            step=300,
            metric_name="val_loss",
            metric_value=2.4,
            mode="min",
        )

        # 1. Verify the targeted query → returns the correct current best checkpoint
        best = manager.best(
            run.run_id,
            "val_loss",
        )

        assert best is not None
        assert best.checkpoint_id == second.checkpoint_id
        assert best.is_best

        # 2. Verify the system invariant: exactly ONE checkpoint is marked best for this metric
        checkpoints = manager.list(run.run_id)
        best_checkpoints = [
            checkpoint
            for checkpoint in checkpoints
            if checkpoint.metric_name == "val_loss" and checkpoint.is_best
        ]

        assert len(best_checkpoints) == 1
        assert best_checkpoints[0].checkpoint_id == second.checkpoint_id

    db.close()


def test_best_checkpoint_with_max_mode(tmp_path):
    db = ResultsDatabase(tmp_path / "results.sqlite")
    manager = CheckpointManager(
        db,
        root_dir=tmp_path / "runs",
    )

    with running_run(db) as run:
        manager.save_bytes(
            run.run_id,
            "model.pt",
            b"first",
            step=100,
            metric_name="accuracy",
            metric_value=0.71,
            mode="max",
        )

        second = manager.save_bytes(
            run.run_id,
            "model.pt",
            b"second",
            step=200,
            metric_name="accuracy",
            metric_value=0.84,
            mode="max",
        )

        # 1. Verify the targeted query returns the correct checkpoint
        best = manager.best(
            run.run_id,
            "accuracy",
        )

        assert best is not None
        assert best.checkpoint_id == second.checkpoint_id
        assert best.is_best

        # 2. Verify the system invariant: exactly ONE checkpoint is marked best for this metric
        checkpoints = manager.list(run.run_id)
        best_checkpoints = [
            checkpoint
            for checkpoint in checkpoints
            if checkpoint.metric_name == "accuracy" and checkpoint.is_best
        ]

        assert len(best_checkpoints) == 1
        assert best_checkpoints[0].checkpoint_id == second.checkpoint_id

    db.close()


def test_checksum_detects_corruption(tmp_path):
    db = ResultsDatabase(tmp_path / "results.sqlite")
    manager = CheckpointManager(
        db,
        root_dir=tmp_path / "runs",
    )

    with running_run(db) as run:
        checkpoint = manager.save_bytes(
            run.run_id,
            "model.pt",
            b"original",
            step=100,
        )

        checkpoint.path.write_bytes(b"corrupted")

        with pytest.raises(CheckpointCorruptionError):
            manager.load(
                checkpoint.checkpoint_id,
                lambda path: path.read_bytes(),
            )

    db.close()


def test_checkpoint_for_completed_run_is_rejected(tmp_path):
    db = ResultsDatabase(tmp_path / "results.sqlite")
    manager = CheckpointManager(
        db,
        root_dir=tmp_path / "runs",
    )

    # 1. Start and immediately finish the run inside the with-block
    with running_run(db) as run:
        completed_run_id = run.run_id
    
    # 2. Now the run is marked 'completed' in the database.
    # We assert that trying to save a checkpoint for it raises a ValueError.
    with pytest.raises(ValueError):
        manager.save_bytes(
            completed_run_id,
            "model.pt",
            b"data",
        )

    db.close()


def test_checkpoint_name_cannot_escape_directory(tmp_path):
    db = ResultsDatabase(tmp_path / "results.sqlite")
    manager = CheckpointManager(
        db,
        root_dir=tmp_path / "runs",
    )

    with running_run(db) as run, pytest.raises(ValueError):
        manager.save_bytes(
            run.run_id,
            "../evil.pt",
            b"malicious",
        )

    db.close()