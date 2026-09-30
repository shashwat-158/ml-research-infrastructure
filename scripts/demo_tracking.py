from pathlib import Path

from ml_infra.config import load_config
from ml_infra.results_db import ResultsDatabase
from ml_infra.tracker import ExperimentTracker


def main() -> None:
    config = load_config(
        "configs/example.yaml"
    )

    db = ResultsDatabase(
        Path("runs") / "results.sqlite"
    )

    tracker = ExperimentTracker(db)

    with tracker.run(
        config,
        repo_root=".",
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