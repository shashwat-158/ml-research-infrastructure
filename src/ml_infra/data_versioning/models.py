from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DatasetFile:
    """
    Immutable description of one file in a dataset version.
    """

    relative_path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class DatasetVersion:
    """
    Immutable snapshot of a registered dataset version.
    """

    dataset_id: str
    version_id: str
    source: str

    manifest: dict[str, Any]
    files: tuple[DatasetFile, ...]

    sample_count: int | None
    schema: dict[str, Any] | None

    created_at: str


class DatasetIntegrityError(RuntimeError):
    """Raised when dataset contents do not match their registered version."""