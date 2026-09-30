import json
import sqlite3
from pathlib import Path
from typing import Any


SCHEMA = """
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    run_name TEXT NOT NULL,

    seed INTEGER NOT NULL,

    config_hash TEXT NOT NULL,
    config_json TEXT NOT NULL,

    git_commit TEXT,
    dataset_version TEXT,

    status TEXT NOT NULL
        CHECK (status IN ('running', 'completed', 'failed')),

    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value REAL NOT NULL,
    step INTEGER,
    timestamp TEXT NOT NULL,

    FOREIGN KEY (run_id)
        REFERENCES runs(run_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS params (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value_json TEXT NOT NULL,

    FOREIGN KEY (run_id)
        REFERENCES runs(run_id)
        ON DELETE CASCADE,

    UNIQUE(run_id, name)
);

CREATE TABLE IF NOT EXISTS tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    run_id TEXT NOT NULL,
    name TEXT NOT NULL,
    value TEXT NOT NULL,

    FOREIGN KEY (run_id)
        REFERENCES runs(run_id)
        ON DELETE CASCADE,

    UNIQUE(run_id, name)
);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,

    run_id TEXT NOT NULL,

    name TEXT NOT NULL,
    path TEXT NOT NULL,
    artifact_type TEXT,

    size_bytes INTEGER,
    sha256 TEXT,

    created_at TEXT NOT NULL,

    FOREIGN KEY (run_id)
        REFERENCES runs(run_id)
        ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_runs_project
    ON runs(project);

CREATE INDEX IF NOT EXISTS idx_runs_status
    ON runs(status);

CREATE INDEX IF NOT EXISTS idx_metrics_run
    ON metrics(run_id);

CREATE INDEX IF NOT EXISTS idx_metrics_name
    ON metrics(name);

CREATE INDEX IF NOT EXISTS idx_artifacts_run
    ON artifacts(run_id);
"""


class ResultsDatabase:
    """
    SQLite-backed persistent store for experiment metadata and results.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

        # ":memory:" (used for fast, temporary in-memory databases during unit testing)
        if self.path != Path(":memory:"):
            # ensures that the folder holding the .sqlite file exists before trying to connect to it
            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._connection = sqlite3.connect(
            self.path,
            # SQLite normally blocks multiple threads from sharing a connection.
            # Disabling this is useful if data loading or evaluation loops run on background threads but need to log metrics.
            check_same_thread=False,
        )

        self._connection.row_factory = sqlite3.Row

        self._connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        self._initialize_schema()

    def _initialize_schema(self) -> None:
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def close(self) -> None:
        """Close the database connection."""

        self._connection.close()

    def __enter__(self) -> "ResultsDatabase":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def create_run(
        self,
        *,
        run_id: str,
        project: str,
        run_name: str,
        seed: int,
        config_hash: str,
        config: dict[str, Any],
        git_commit: str | None,
        dataset_version: str | None,
        started_at: str,
        tags: dict[str, str],
    ) -> None:
        """
        Create a new experiment run.
        """

        config_json = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
        )

        with self._connection:
            self._connection.execute(
                """
                INSERT INTO runs (
                    run_id,
                    project,
                    run_name,
                    seed,
                    config_hash,
                    config_json,
                    git_commit,
                    dataset_version,
                    status,
                    started_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    run_id,
                    project,
                    run_name,
                    seed,
                    config_hash,
                    config_json,
                    git_commit,
                    dataset_version,
                    started_at,
                ),
            )

            for name, value in tags.items():
                self._connection.execute(
                    """
                    INSERT INTO tags (
                        run_id,
                        name,
                        value
                    )
                    VALUES (?, ?, ?)
                    """,
                    (run_id, name, value),
                )

    def complete_run(
        self,
        run_id: str,
        ended_at: str,
    ) -> None:
        """Mark a run as successfully completed."""

        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE runs
                SET status = 'completed',
                    ended_at = ?
                WHERE run_id = ?
                  AND status = 'running'
                """,
                (ended_at, run_id),
            )

        if cursor.rowcount != 1:
            raise ValueError(
                f"Cannot complete run '{run_id}'. "
                "Run may not exist or may already be finished."
            )

    def fail_run(
        self,
        run_id: str,
        ended_at: str,
    ) -> None:
        """Mark a run as failed."""

        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE runs
                SET status = 'failed',
                    ended_at = ?
                WHERE run_id = ?
                  AND status = 'running'
                """,
                (ended_at, run_id),
            )

        if cursor.rowcount != 1:
            raise ValueError(
                f"Cannot fail run '{run_id}'. "
                "Run may not exist or may already be finished."
            )

    def log_metric(
        self,
        *,
        run_id: str,
        name: str,
        value: float,
        step: int | None,
        timestamp: str,
    ) -> None:
        """Store one metric observation."""

        with self._connection:
            self._connection.execute(
                """
                INSERT INTO metrics (
                    run_id,
                    name,
                    value,
                    step,
                    timestamp
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    name,
                    value,
                    step,
                    timestamp,
                ),
            )

    def log_param(
        self,
        *,
        run_id: str,
        name: str,
        value: Any,
    ) -> None:
        """Store one parameter value as JSON."""

        value_json = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        )

        with self._connection:
            self._connection.execute(
                """
                INSERT INTO params (
                    run_id,
                    name,
                    value_json
                )
                VALUES (?, ?, ?)
                ON CONFLICT(run_id, name)
                DO UPDATE SET value_json = excluded.value_json
                """,
                (
                    run_id,
                    name,
                    value_json,
                ),
            )

    def register_artifact(
        self,
        *,
        run_id: str,
        name: str,
        path: str,
        artifact_type: str | None,
        size_bytes: int | None,
        sha256: str | None,
        created_at: str,
    ) -> None:
        """Register an artifact produced by a run."""

        with self._connection:
            self._connection.execute(
                """
                INSERT INTO artifacts (
                    run_id,
                    name,
                    path,
                    artifact_type,
                    size_bytes,
                    sha256,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    name,
                    path,
                    artifact_type,
                    size_bytes,
                    sha256,
                    created_at,
                ),
            )

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one run as a dictionary."""

        row = self._connection.execute(
            """
            SELECT *
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()

        if row is None:
            return None

        result = dict(row)

        result["config"] = json.loads(
            result.pop("config_json")
        )

        return result

    def get_metrics(
        self,
        run_id: str,
        name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return metrics for a run."""

        if name is None:
            rows = self._connection.execute(
                """
                SELECT name, value, step, timestamp
                FROM metrics
                WHERE run_id = ?
                ORDER BY id
                """,
                (run_id,),
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT name, value, step, timestamp
                FROM metrics
                WHERE run_id = ?
                  AND name = ?
                ORDER BY id
                """,
                (run_id, name),
            ).fetchall()

        return [dict(row) for row in rows]

    def list_runs(
        self,
        project: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return experiment runs, optionally filtered by project."""

        if project is None:
            rows = self._connection.execute(
                """
                SELECT
                    run_id,
                    project,
                    run_name,
                    seed,
                    config_hash,
                    git_commit,
                    dataset_version,
                    status,
                    started_at,
                    ended_at
                FROM runs
                ORDER BY started_at DESC
                """
            ).fetchall()
        else:
            rows = self._connection.execute(
                """
                SELECT
                    run_id,
                    project,
                    run_name,
                    seed,
                    config_hash,
                    git_commit,
                    dataset_version,
                    status,
                    started_at,
                    ended_at
                FROM runs
                WHERE project = ?
                ORDER BY started_at DESC
                """,
                (project,),
            ).fetchall()

        return [dict(row) for row in rows]