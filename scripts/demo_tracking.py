from pathlib import Path

from ml_infra.config import load_config
from ml_infra.data_versioning import (
    DatasetVersionManager,
)
from ml_infra.results_db import ResultsDatabase
from ml_infra.tracker import ExperimentTracker


def main() -> None:
    config = load_config(
        "configs/example.yaml"
    )

    db = ResultsDatabase(
        Path("runs") / "results.sqlite"
    )

    dataset_path = Path(
        "runs/demo_dataset.txt"
    )

    dataset_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataset_path.write_text(
        "example dataset\n",
        encoding="utf-8",
    )

    dataset_manager = (
        DatasetVersionManager(db)
    )

    dataset_version = (
        dataset_manager.register(
            dataset_id="demo-dataset",
            path=dataset_path,
            source="local://demo-dataset",
            sample_count=1,
            schema={
                "content": "text",
            },
        )
    )

    tracker = ExperimentTracker(db)

    with tracker.run(
        config,
        repo_root=".",
        dataset_version=dataset_version.version_id,
    ) as run:

        for step, loss in enumerate(
            [4.2, 3.7, 3.1, 2.8, 2.4],
            start=1,
        ):
            run.log_metric(
                "train_loss",
                loss,
                step=step,
            )

        run.log_param(
            "optimizer",
            "AdamW",
        )

    print(
        f"Experiment completed: {run.run_id}"
    )

    db.close()


if __name__ == "__main__":
    main()