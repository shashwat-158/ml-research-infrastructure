from pathlib import Path

import pytest

from ml_infra.config import RunConfig, config_hash, load_config


def test_load_config(tmp_path: Path) -> None:
    config_file = tmp_path / "config.yaml"

    config_file.write_text(
        """
project: test-project
run_name: baseline
seed: 42
output_dir: runs
parameters:
  learning_rate: 0.001
  batch_size: 32
tags:
  stage: test
""",
        encoding="utf-8",
    )

    config = load_config(config_file)

    assert config.project == "test-project"
    assert config.run_name == "baseline"
    assert config.seed == 42
    assert config.parameters["learning_rate"] == 0.001
    assert config.parameters["batch_size"] == 32
    assert config.tags["stage"] == "test"


def test_config_hash_is_deterministic() -> None:
    config_a = RunConfig(
        project="test-project",
        run_name="baseline",
        seed=42,
        parameters={
            "learning_rate": 0.001,
            "batch_size": 32,
        },
    )

    config_b = RunConfig(
        project="test-project",
        run_name="baseline",
        seed=42,
        parameters={
            "batch_size": 32,
            "learning_rate": 0.001,
        },
    )

    assert config_hash(config_a) == config_hash(config_b)


def test_config_hash_changes_when_experiment_changes() -> None:
    config_a = RunConfig(
        project="test-project",
        run_name="baseline",
        seed=42,
    )

    config_b = RunConfig(
        project="test-project",
        run_name="baseline",
        seed=43,
    )

    assert config_hash(config_a) != config_hash(config_b)


def test_unknown_top_level_field_is_rejected() -> None:
    with pytest.raises(ValueError):
        RunConfig(
            project="test-project",
            run_name="baseline",
            unexpected_field="bad",
        )


def test_missing_config_file_is_rejected() -> None:
    with pytest.raises(FileNotFoundError):
        load_config("does-not-exist.yaml")