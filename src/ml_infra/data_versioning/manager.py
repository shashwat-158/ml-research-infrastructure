import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ml_infra.results_db import ResultsDatabase

from .models import (
    DatasetFile,
    DatasetIntegrityError,
    DatasetVersion,
)


class DatasetVersionManager:
    """
    Register and verify immutable dataset versions.
    """

    def __init__(
        self,
        db: ResultsDatabase,
    ):
        self.db = db

    @staticmethod
    def _utc_now() -> str:
        """Return current UTC timestamp."""

        return datetime.now(
            timezone.utc
        ).isoformat()

    @staticmethod
    def _sha256_file(
        path: Path,
    ) -> str:
        """Calculate SHA-256 incrementally."""

        digest = hashlib.sha256()

        with path.open("rb") as file:
            while chunk := file.read(
                1024 * 1024
            ):
                digest.update(chunk)

        return digest.hexdigest()

    @classmethod
    def _discover_files(
        cls,
        path: Path,
    ) -> tuple[str, list[DatasetFile]]:
        """
        Discover all files contributing to the dataset.

        Returns:
            layout type and sorted file records.
        """

        if path.is_symlink():
            raise ValueError(
                "Dataset root cannot be a symbolic link."
            )

        if path.is_file():
            file_record = DatasetFile(
                relative_path=path.name,
                size_bytes=path.stat().st_size,
                sha256=cls._sha256_file(path),
            )

            return "single_file", [
                file_record
            ]

        if not path.is_dir():
            raise ValueError(
                f"Dataset path does not exist "
                f"or is not a file/directory: {path}"
            )

        records: list[DatasetFile] = []

        for child in path.rglob("*"):

            if child.is_symlink():
                raise ValueError(
                    "Dataset contains a symbolic link: "
                    f"{child}"
                )

            if not child.is_file():
                continue

            relative_path = child.relative_to(
                path
            ).as_posix()

            records.append(
                DatasetFile(
                    relative_path=relative_path,
                    size_bytes=child.stat().st_size,
                    sha256=cls._sha256_file(child),
                )
            )

        records.sort(
            key=lambda item: item.relative_path
        )

        if not records:
            raise ValueError(
                f"Dataset directory is empty: {path}"
            )

        return "directory", records

    @staticmethod
    def _build_manifest(
        dataset_id: str,
        layout: str,
        files: list[DatasetFile],
    ) -> dict[str, Any]:
        """
        Build the canonical identity manifest.

        Only dataset identity/layout and file content identity
        participate in the version hash.
        """

        return {
            "dataset_id": dataset_id,
            "layout": layout,
            "files": [
                {
                    "relative_path": file.relative_path,
                    "size_bytes": file.size_bytes,
                    "sha256": file.sha256,
                }
                for file in files
            ],
        }

    @staticmethod
    def _version_id(
        manifest: dict[str, Any],
    ) -> str:
        """
        Compute a deterministic SHA-256 identity from the
        canonical manifest.
        """

        canonical_json = json.dumps(
            manifest,
            sort_keys=True,
            separators=(",", ":"),
        )

        return hashlib.sha256(
            canonical_json.encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _build_version(
        record: dict[str, Any],
        file_records: list[dict[str, Any]],
    ) -> DatasetVersion:

        files = tuple(
            DatasetFile(
                relative_path=file["relative_path"],
                size_bytes=file["size_bytes"],
                sha256=file["sha256"],
            )
            for file in file_records
        )

        return DatasetVersion(
            dataset_id=record["dataset_id"],
            version_id=record["version_id"],
            source=record["source"],
            manifest=record["manifest"],
            files=files,
            sample_count=record["sample_count"],
            schema=record["schema"],
            created_at=record["created_at"],
        )

    def register(
        self,
        *,
        dataset_id: str,
        path: str | Path,
        source: str,
        sample_count: int | None = None,
        schema: dict[str, Any] | None = None,
    ) -> DatasetVersion:
        """
        Register dataset contents as an immutable version.

        The version ID depends only on dataset identity/layout
        and the discovered file contents.
        """

        if not dataset_id:
            raise ValueError(
                "dataset_id cannot be empty."
            )

        if not source:
            raise ValueError(
                "source cannot be empty."
            )

        if sample_count is not None:
            if not isinstance(sample_count, int):
                raise TypeError(
                    "sample_count must be an integer."
                )

            if sample_count < 0:
                raise ValueError(
                    "sample_count cannot be negative."
                )

        if schema is not None:
            if not isinstance(schema, dict):
                raise TypeError(
                    "schema must be a dictionary."
                )

        path = Path(path)

        layout, files = self._discover_files(
            path
        )

        manifest = self._build_manifest(
            dataset_id=dataset_id,
            layout=layout,
            files=files,
        )

        version_id = self._version_id(
            manifest
        )

        self.db.create_dataset_version(
            version_id=version_id,
            dataset_id=dataset_id,
            source=source,
            manifest=manifest,
            sample_count=sample_count,
            schema=schema,
            created_at=self._utc_now(),
            files=[
                {
                    "relative_path": file.relative_path,
                    "size_bytes": file.size_bytes,
                    "sha256": file.sha256,
                }
                for file in files
            ],
        )

        return self.get(version_id)

    def get(
        self,
        version_id: str,
    ) -> DatasetVersion:

        record = self.db.get_dataset_version(
            version_id
        )

        if record is None:
            raise ValueError(
                f"Dataset version '{version_id}' "
                "does not exist."
            )

        file_records = (
            self.db.get_dataset_files(
                version_id
            )
        )

        return self._build_version(
            record,
            file_records,
        )

    def list(
        self,
        dataset_id: str,
    ) -> list[DatasetVersion]:

        records = (
            self.db.list_dataset_versions(
                dataset_id
            )
        )

        versions: list[DatasetVersion] = []

        for record in records:
            file_records = (
                self.db.get_dataset_files(
                    record["version_id"]
                )
            )

            # The list query does not include the
            # manifest/schema, so retrieve the full record.
            full_record = (
                self.db.get_dataset_version(
                    record["version_id"]
                )
            )

            if full_record is None:
                raise RuntimeError(
                    "Dataset version disappeared while "
                    "loading version list."
                )

            versions.append(
                self._build_version(
                    full_record,
                    file_records,
                )
            )

        return versions

    def verify(
        self,
        version_id: str,
        path: str | Path,
    ) -> None:
        """
        Verify that current dataset contents match the
        registered version.

        Raises DatasetIntegrityError on mismatch.
        """

        expected = self.get(
            version_id
        )

        path = Path(path)

        layout, files = (
            self._discover_files(path)
        )

        current_manifest = (
            self._build_manifest(
                dataset_id=expected.dataset_id,
                layout=layout,
                files=files,
            )
        )

        current_version_id = (
            self._version_id(
                current_manifest
            )
        )

        if current_version_id != version_id:
            raise DatasetIntegrityError(
                "Dataset contents do not match "
                f"version '{version_id}'. "
                f"Expected version: {version_id}; "
                f"actual version: {current_version_id}."
            )