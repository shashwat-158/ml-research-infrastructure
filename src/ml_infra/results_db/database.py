import json
import sqlite3
from pathlib import Path
from typing import Any, Self

CURRENT_SCHEMA_VERSION = 4


SCHEMA_V1 = """
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


MIGRATIONS = {
    2: """
    CREATE TABLE IF NOT EXISTS checkpoints (
        checkpoint_id TEXT PRIMARY KEY,

        run_id TEXT NOT NULL,

        name TEXT NOT NULL,
        path TEXT NOT NULL,

        step INTEGER,

        metric_name TEXT,
        metric_value REAL,

        is_latest INTEGER NOT NULL DEFAULT 0
            CHECK (is_latest IN (0, 1)),

        is_best INTEGER NOT NULL DEFAULT 0
            CHECK (is_best IN (0, 1)),

        size_bytes INTEGER NOT NULL,

        sha256 TEXT NOT NULL,

        created_at TEXT NOT NULL,

        FOREIGN KEY (run_id)
            REFERENCES runs(run_id)
            ON DELETE CASCADE,

        CHECK (
            is_best = 0
            OR metric_name IS NOT NULL
        )
    );

    CREATE INDEX IF NOT EXISTS idx_checkpoints_run
        ON checkpoints(run_id);

    CREATE INDEX IF NOT EXISTS idx_checkpoints_metric
        ON checkpoints(run_id, metric_name);

    CREATE UNIQUE INDEX IF NOT EXISTS idx_one_latest_checkpoint
        ON checkpoints(run_id)
        WHERE is_latest = 1;

    CREATE UNIQUE INDEX IF NOT EXISTS idx_one_best_checkpoint_per_metric
        ON checkpoints(run_id, metric_name)
        WHERE is_best = 1;
    """,
    3: """
    CREATE TABLE IF NOT EXISTS datasets (
        dataset_id TEXT PRIMARY KEY,

        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS dataset_versions (
        version_id TEXT PRIMARY KEY,

        dataset_id TEXT NOT NULL,

        source TEXT NOT NULL,

        manifest_json TEXT NOT NULL,

        sample_count INTEGER,

        schema_json TEXT,

        created_at TEXT NOT NULL,

        FOREIGN KEY (dataset_id)
            REFERENCES datasets(dataset_id)
            ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS dataset_files (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        version_id TEXT NOT NULL,

        relative_path TEXT NOT NULL,

        size_bytes INTEGER NOT NULL,

        sha256 TEXT NOT NULL,

        FOREIGN KEY (version_id)
            REFERENCES dataset_versions(version_id)
            ON DELETE CASCADE,

        UNIQUE(version_id, relative_path)
    );

    CREATE INDEX IF NOT EXISTS idx_dataset_versions_dataset
        ON dataset_versions(dataset_id);

    CREATE INDEX IF NOT EXISTS idx_dataset_files_version
        ON dataset_files(version_id);

    CREATE INDEX IF NOT EXISTS idx_dataset_files_sha256
        ON dataset_files(sha256);
    """,
        4: """
    CREATE TABLE IF NOT EXISTS evaluation_runs (
        evaluation_id TEXT PRIMARY KEY,

        run_id TEXT,

        suite_name TEXT NOT NULL,
        suite_hash TEXT NOT NULL,

        evaluator_names_json TEXT NOT NULL,
        config_json TEXT NOT NULL,

        seed INTEGER,

        status TEXT NOT NULL
            CHECK (
                status IN (
                    'running',
                    'completed',
                    'failed'
                )
            ),

        started_at TEXT NOT NULL,
        ended_at TEXT,

        FOREIGN KEY (run_id)
            REFERENCES runs(run_id)
            ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS evaluation_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        evaluation_id TEXT NOT NULL,

        case_id TEXT NOT NULL,
        evaluator_name TEXT NOT NULL,

        passed INTEGER
            CHECK (
                passed IS NULL
                OR passed IN (0, 1)
            ),

        score REAL,

        explanation TEXT,

        details_json TEXT NOT NULL,

        duration_ms REAL NOT NULL,

        created_at TEXT NOT NULL,

        FOREIGN KEY (evaluation_id)
            REFERENCES evaluation_runs(evaluation_id)
            ON DELETE CASCADE,

        UNIQUE(
            evaluation_id,
            case_id,
            evaluator_name
        )
    );

    CREATE TABLE IF NOT EXISTS evaluation_metrics (
        id INTEGER PRIMARY KEY AUTOINCREMENT,

        evaluation_id TEXT NOT NULL,

        evaluator_name TEXT NOT NULL,
        name TEXT NOT NULL,

        value REAL NOT NULL,

        created_at TEXT NOT NULL,

        FOREIGN KEY (evaluation_id)
            REFERENCES evaluation_runs(evaluation_id)
            ON DELETE CASCADE,

        UNIQUE(
            evaluation_id,
            evaluator_name,
            name
        )
    );

    CREATE INDEX IF NOT EXISTS idx_evaluation_runs_run
        ON evaluation_runs(run_id);

    CREATE INDEX IF NOT EXISTS idx_evaluation_runs_suite
        ON evaluation_runs(suite_name);

    CREATE INDEX IF NOT EXISTS idx_evaluation_results_evaluation
        ON evaluation_results(evaluation_id);

    CREATE INDEX IF NOT EXISTS idx_evaluation_results_case
        ON evaluation_results(case_id);

    CREATE INDEX IF NOT EXISTS idx_evaluation_results_evaluator
        ON evaluation_results(evaluator_name);

    CREATE INDEX IF NOT EXISTS idx_evaluation_metrics_evaluation
        ON evaluation_metrics(evaluation_id);
    """
}


class ResultsDatabase:
    """
    SQLite-backed persistent store for experiment metadata and results.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)

        if self.path != Path(":memory:"):
            self.path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

        self._connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
        )

        self._connection.row_factory = sqlite3.Row

        self._connection.execute(
            "PRAGMA foreign_keys = ON"
        )

        self._initialize_schema()

    def _initialize_schema(self) -> None:
        """
        Initialize the database and apply all required migrations.
        """

        with self._connection:
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_version (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    version INTEGER NOT NULL
                )
                """
            )

            row = self._connection.execute(
                """
                SELECT version
                FROM schema_version
                WHERE id = 1
                """
            ).fetchone()

            if row is None:

                runs_exists = self._connection.execute(
                    """
                    SELECT 1
                    FROM sqlite_master
                    WHERE type = 'table'
                      AND name = 'runs'
                    """
                ).fetchone()

                if runs_exists is None:
                    self._connection.executescript(
                        SCHEMA_V1
                    )

                self._connection.execute(
                    """
                    INSERT INTO schema_version (
                        id,
                        version
                    )
                    VALUES (1, 1)
                    """
                )

                current_version = 1

            else:
                current_version = row["version"]

            if current_version > CURRENT_SCHEMA_VERSION:
                raise RuntimeError(
                    "Database schema version "
                    f"{current_version} is newer than the "
                    f"supported version {CURRENT_SCHEMA_VERSION}."
                )

            while current_version < CURRENT_SCHEMA_VERSION:
                next_version = current_version + 1

                migration = MIGRATIONS.get(
                    next_version
                )

                if migration is None:
                    raise RuntimeError(
                        f"Missing migration for schema version "
                        f"{next_version}."
                    )

                self._connection.executescript(
                    migration
                )

                self._connection.execute(
                    """
                    UPDATE schema_version
                    SET version = ?
                    WHERE id = 1
                    """,
                    (next_version,),
                )

                current_version = next_version

    def schema_version(self) -> int:
        """Return the current database schema version."""

        row = self._connection.execute(
            """
            SELECT version
            FROM schema_version
            WHERE id = 1
            """
        ).fetchone()

        if row is None:
            raise RuntimeError(
                "Database schema version is missing."
            )

        return row["version"]

    def close(self) -> None:
        """Close the database connection."""

        self._connection.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
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

        if dataset_version is not None:
            if not self.dataset_version_exists(
                dataset_version
            ):
                raise ValueError(
                    f"Dataset version '{dataset_version}' "
                    "does not exist."
                )

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
                    (
                        run_id,
                        name,
                        value,
                    ),
                )

    def complete_run(
        self,
        run_id: str,
        ended_at: str,
    ) -> None:

        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE runs
                SET status = 'completed',
                    ended_at = ?
                WHERE run_id = ?
                  AND status = 'running'
                """,
                (
                    ended_at,
                    run_id,
                ),
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

        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE runs
                SET status = 'failed',
                    ended_at = ?
                WHERE run_id = ?
                  AND status = 'running'
                """,
                (
                    ended_at,
                    run_id,
                ),
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

    def get_run(
        self,
        run_id: str,
    ) -> dict[str, Any] | None:

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
                (
                    run_id,
                    name,
                ),
            ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def list_runs(
        self,
        project: str | None = None,
    ) -> list[dict[str, Any]]:

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

        return [
            dict(row)
            for row in rows
        ]

    # ------------------------------------------------------------------
    # Checkpoint operations
    # ------------------------------------------------------------------

    def register_checkpoint(
        self,
        *,
        checkpoint_id: str,
        run_id: str,
        name: str,
        path: str,
        step: int | None,
        metric_name: str | None,
        metric_value: float | None,
        is_latest: bool,
        is_best: bool,
        size_bytes: int,
        sha256: str,
        created_at: str,
    ) -> None:
        """
        Register a checkpoint and update latest/best pointers atomically.
        """

        with self._connection:
            if is_latest:
                self._connection.execute(
                    """
                    UPDATE checkpoints
                    SET is_latest = 0
                    WHERE run_id = ?
                    """,
                    (run_id,),
                )

            if is_best:
                self._connection.execute(
                    """
                    UPDATE checkpoints
                    SET is_best = 0
                    WHERE run_id = ?
                      AND metric_name = ?
                    """,
                    (
                        run_id,
                        metric_name,
                    ),
                )

            self._connection.execute(
                """
                INSERT INTO checkpoints (
                    checkpoint_id,
                    run_id,
                    name,
                    path,
                    step,
                    metric_name,
                    metric_value,
                    is_latest,
                    is_best,
                    size_bytes,
                    sha256,
                    created_at
                )
                VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
                """,
                (
                    checkpoint_id,
                    run_id,
                    name,
                    path,
                    step,
                    metric_name,
                    metric_value,
                    int(is_latest),
                    int(is_best),
                    size_bytes,
                    sha256,
                    created_at,
                ),
            )

    def get_checkpoint(
        self,
        checkpoint_id: str,
    ) -> dict[str, Any] | None:

        row = self._connection.execute(
            """
            SELECT *
            FROM checkpoints
            WHERE checkpoint_id = ?
            """,
            (checkpoint_id,),
        ).fetchone()

        if row is None:
            return None

        return dict(row)

    def get_latest_checkpoint(
        self,
        run_id: str,
    ) -> dict[str, Any] | None:

        row = self._connection.execute(
            """
            SELECT *
            FROM checkpoints
            WHERE run_id = ?
              AND is_latest = 1
            """,
            (run_id,),
        ).fetchone()

        if row is None:
            return None

        return dict(row)

    def get_best_checkpoint(
        self,
        run_id: str,
        metric_name: str,
    ) -> dict[str, Any] | None:

        row = self._connection.execute(
            """
            SELECT *
            FROM checkpoints
            WHERE run_id = ?
              AND metric_name = ?
              AND is_best = 1
            """,
            (
                run_id,
                metric_name,
            ),
        ).fetchone()

        if row is None:
            return None

        return dict(row)

    def list_checkpoints(
        self,
        run_id: str,
    ) -> list[dict[str, Any]]:

        rows = self._connection.execute(
            """
            SELECT *
            FROM checkpoints
            WHERE run_id = ?
            ORDER BY created_at
            """,
            (run_id,),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def get_artifacts(
        self,
        run_id: str,
    ) -> list[dict[str, Any]]:
        """Return all artifacts registered for a run."""

        rows = self._connection.execute(
            """
            SELECT
                id,
                run_id,
                name,
                path,
                artifact_type,
                size_bytes,
                sha256,
                created_at
            FROM artifacts
            WHERE run_id = ?
            ORDER BY id
            """,
            (run_id,),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def create_dataset(
        self,
        *,
        dataset_id: str,
        created_at: str,
    ) -> None:
        """Create a logical dataset if it does not already exist."""

        with self._connection:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO datasets (
                    dataset_id,
                    created_at
                )
                VALUES (?, ?)
                """,
                (
                    dataset_id,
                    created_at,
                ),
            )

    def create_dataset_version(
        self,
        *,
        version_id: str,
        dataset_id: str,
        source: str,
        manifest: dict[str, Any],
        sample_count: int | None,
        schema: dict[str, Any] | None,
        created_at: str,
        files: list[dict[str, Any]],
    ) -> None:
        """
        Persist an immutable dataset version and its file manifest.

        Re-registering an existing version is a no-op.
        """

        manifest_json = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
        )

        schema_json = (
            json.dumps(
                schema,
                sort_keys=True,
                separators=(",", ":"),
            )
            if schema is not None
            else None
        )

        with self._connection:
            self._connection.execute(
                """
                INSERT OR IGNORE INTO datasets (
                    dataset_id,
                    created_at
                )
                VALUES (?, ?)
                """,
                (
                    dataset_id,
                    created_at,
                ),
            )

            existing = self._connection.execute(
                """
                SELECT version_id
                FROM dataset_versions
                WHERE version_id = ?
                """,
                (version_id,),
            ).fetchone()

            if existing is not None:
                return

            self._connection.execute(
                """
                INSERT INTO dataset_versions (
                    version_id,
                    dataset_id,
                    source,
                    manifest_json,
                    sample_count,
                    schema_json,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    version_id,
                    dataset_id,
                    source,
                    manifest_json,
                    sample_count,
                    schema_json,
                    created_at,
                ),
            )

            for file_record in files:
                self._connection.execute(
                    """
                    INSERT INTO dataset_files (
                        version_id,
                        relative_path,
                        size_bytes,
                        sha256
                    )
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        version_id,
                        file_record["relative_path"],
                        file_record["size_bytes"],
                        file_record["sha256"],
                    ),
                )

    def get_dataset_version(
        self,
        version_id: str,
    ) -> dict[str, Any] | None:
        """Return one dataset version."""

        row = self._connection.execute(
            """
            SELECT
                version_id,
                dataset_id,
                source,
                manifest_json,
                sample_count,
                schema_json,
                created_at
            FROM dataset_versions
            WHERE version_id = ?
            """,
            (version_id,),
        ).fetchone()

        if row is None:
            return None

        result = dict(row)

        result["manifest"] = json.loads(
            result.pop("manifest_json")
        )

        schema_json = result.pop(
            "schema_json"
        )

        result["schema"] = (
            json.loads(schema_json)
            if schema_json is not None
            else None
        )

        return result

    def get_dataset_files(
        self,
        version_id: str,
    ) -> list[dict[str, Any]]:
        """Return all files belonging to a dataset version."""

        rows = self._connection.execute(
            """
            SELECT
                relative_path,
                size_bytes,
                sha256
            FROM dataset_files
            WHERE version_id = ?
            ORDER BY relative_path
            """,
            (version_id,),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def list_dataset_versions(
        self,
        dataset_id: str,
    ) -> list[dict[str, Any]]:
        """Return all versions belonging to a dataset."""

        rows = self._connection.execute(
            """
            SELECT
                version_id,
                dataset_id,
                source,
                sample_count,
                created_at
            FROM dataset_versions
            WHERE dataset_id = ?
            ORDER BY created_at
            """,
            (dataset_id,),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]

    def dataset_version_exists(
        self,
        version_id: str,
    ) -> bool:
        """Return whether a dataset version exists."""

        row = self._connection.execute(
            """
            SELECT 1
            FROM dataset_versions
            WHERE version_id = ?
            """,
            (version_id,),
        ).fetchone()

        return row is not None

    def create_evaluation(
        self,
        *,
        evaluation_id: str,
        run_id: str | None,
        suite_name: str,
        suite_hash: str,
        evaluator_names: list[str],
        config: dict[str, Any],
        seed: int | None,
        started_at: str,
    ) -> None:
        """Create a running evaluation record."""

        if run_id is not None:
            run = self.get_run(run_id)

            if run is None:
                raise ValueError(
                    f"Run '{run_id}' does not exist."
                )

        evaluator_names_json = json.dumps(
            evaluator_names,
            sort_keys=True,
            separators=(",", ":"),
        )

        config_json = json.dumps(
            config,
            sort_keys=True,
            separators=(",", ":"),
        )

        with self._connection:
            self._connection.execute(
                """
                INSERT INTO evaluation_runs (
                    evaluation_id,
                    run_id,
                    suite_name,
                    suite_hash,
                    evaluator_names_json,
                    config_json,
                    seed,
                    status,
                    started_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'running', ?)
                """,
                (
                    evaluation_id,
                    run_id,
                    suite_name,
                    suite_hash,
                    evaluator_names_json,
                    config_json,
                    seed,
                    started_at,
                ),
            )

    def complete_evaluation(
        self,
        evaluation_id: str,
        ended_at: str,
    ) -> None:
        """Mark an evaluation as completed."""

        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE evaluation_runs
                SET status = 'completed',
                    ended_at = ?
                WHERE evaluation_id = ?
                AND status = 'running'
                """,
                (
                    ended_at,
                    evaluation_id,
                ),
            )

        if cursor.rowcount != 1:
            raise ValueError(
                f"Cannot complete evaluation "
                f"'{evaluation_id}'."
            )


    def fail_evaluation(
        self,
        evaluation_id: str,
        ended_at: str,
    ) -> None:
        """Mark an evaluation as failed."""

        with self._connection:
            cursor = self._connection.execute(
                """
                UPDATE evaluation_runs
                SET status = 'failed',
                    ended_at = ?
                WHERE evaluation_id = ?
                AND status = 'running'
                """,
                (
                    ended_at,
                    evaluation_id,
                ),
            )

        if cursor.rowcount != 1:
            raise ValueError(
                f"Cannot fail evaluation "
                f"'{evaluation_id}'."
            )

    def log_evaluation_result(
        self,
        *,
        evaluation_id: str,
        case_id: str,
        evaluator_name: str,
        passed: bool | None,
        score: float | None,
        explanation: str | None,
        details: dict[str, Any],
        duration_ms: float,
        created_at: str,
    ) -> None:
        """Persist one case/evaluator result."""

        passed_value = (
            None
            if passed is None
            else int(passed)
        )

        details_json = json.dumps(
            details,
            sort_keys=True,
            separators=(",", ":"),
        )

        with self._connection:
            self._connection.execute(
                """
                INSERT INTO evaluation_results (
                    evaluation_id,
                    case_id,
                    evaluator_name,
                    passed,
                    score,
                    explanation,
                    details_json,
                    duration_ms,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    evaluation_id,
                    case_id,
                    evaluator_name,
                    passed_value,
                    score,
                    explanation,
                    details_json,
                    duration_ms,
                    created_at,
                ),
            )

    def log_evaluation_metric(
        self,
        *,
        evaluation_id: str,
        evaluator_name: str,
        name: str,
        value: float,
        created_at: str,
    ) -> None:
        """Persist one aggregate evaluation metric."""

        with self._connection:
            self._connection.execute(
                """
                INSERT INTO evaluation_metrics (
                    evaluation_id,
                    evaluator_name,
                    name,
                    value,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(
                    evaluation_id,
                    evaluator_name,
                    name
                )
                DO UPDATE SET value = excluded.value
                """,
                (
                    evaluation_id,
                    evaluator_name,
                    name,
                    value,
                    created_at,
                ),
            )

    def get_evaluation(
        self,
        evaluation_id: str,
    ) -> dict[str, Any] | None:
        """Return one evaluation record."""

        row = self._connection.execute(
            """
            SELECT *
            FROM evaluation_runs
            WHERE evaluation_id = ?
            """,
            (evaluation_id,),
        ).fetchone()

        if row is None:
            return None

        result = dict(row)

        result["evaluator_names"] = json.loads(
            result.pop("evaluator_names_json")
        )

        result["config"] = json.loads(
            result.pop("config_json")
        )

        return result

    def get_evaluation_results(
        self,
        evaluation_id: str,
    ) -> list[dict[str, Any]]:
        """Return all case-level evaluation results."""

        rows = self._connection.execute(
            """
            SELECT
                id,
                evaluation_id,
                case_id,
                evaluator_name,
                passed,
                score,
                explanation,
                details_json,
                duration_ms,
                created_at
            FROM evaluation_results
            WHERE evaluation_id = ?
            ORDER BY id
            """,
            (evaluation_id,),
        ).fetchall()

        results = []

        for row in rows:
            result = dict(row)

            if result["passed"] is not None:
                result["passed"] = bool(
                    result["passed"]
                )

            result["details"] = json.loads(
                result.pop("details_json")
            )

            results.append(result)

        return results

    def get_evaluation_metrics(
        self,
        evaluation_id: str,
    ) -> list[dict[str, Any]]:
        """Return aggregate evaluation metrics."""

        rows = self._connection.execute(
            """
            SELECT
                evaluator_name,
                name,
                value,
                created_at
            FROM evaluation_metrics
            WHERE evaluation_id = ?
            ORDER BY id
            """,
            (evaluation_id,),
        ).fetchall()

        return [
            dict(row)
            for row in rows
        ]