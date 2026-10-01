from pathlib import Path

import pytest

from ml_infra.data_versioning import (
    DatasetIntegrityError,
    DatasetVersionManager,
)
from ml_infra.results_db import ResultsDatabase


def test_register_single_file_dataset(
    tmp_path: Path,
) -> None:

    dataset = (
        tmp_path / "train.csv"
    )

    dataset.write_text(
        "x,y\n1,2\n3,4\n",
        encoding="utf-8",
    )

    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    manager = DatasetVersionManager(db)

    version = manager.register(
        dataset_id="toy",
        path=dataset,
        source="local://toy/train.csv",
        sample_count=2,
        schema={
            "x": "int",
            "y": "int",
        },
    )

    assert version.dataset_id == "toy"
    assert len(version.version_id) == 64
    assert version.source == (
        "local://toy/train.csv"
    )
    assert version.sample_count == 2
    assert version.schema == {
        "x": "int",
        "y": "int",
    }

    assert len(version.files) == 1
    assert version.files[0].relative_path == (
        "train.csv"
    )
    assert version.files[0].size_bytes > 0
    assert len(version.files[0].sha256) == 64

    db.close()


def test_register_directory_creates_deterministic_version(
    tmp_path: Path,
) -> None:

    dataset = tmp_path / "dataset"

    dataset.mkdir()

    (dataset / "train.csv").write_text(
        "a,b\n1,2\n",
        encoding="utf-8",
    )

    (dataset / "test.csv").write_text(
        "a,b\n3,4\n",
        encoding="utf-8",
    )

    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    manager = DatasetVersionManager(db)

    first = manager.register(
        dataset_id="toy",
        path=dataset,
        source="local://toy",
    )

    second = manager.register(
        dataset_id="toy",
        path=dataset,
        source="local://toy",
    )

    assert (
        first.version_id
        == second.version_id
    )

    assert len(first.files) == 2

    assert [
        file.relative_path
        for file in first.files
    ] == [
        "test.csv",
        "train.csv",
    ]

    db.close()


def test_content_change_creates_new_version(
    tmp_path: Path,
) -> None:

    dataset = tmp_path / "dataset"

    dataset.mkdir()

    file = dataset / "train.csv"

    file.write_text(
        "x\n1\n2\n",
        encoding="utf-8",
    )

    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    manager = DatasetVersionManager(db)

    first = manager.register(
        dataset_id="toy",
        path=dataset,
        source="local://toy",
    )

    file.write_text(
        "x\n1\n2\n3\n",
        encoding="utf-8",
    )

    second = manager.register(
        dataset_id="toy",
        path=dataset,
        source="local://toy",
    )

    assert (
        first.version_id
        != second.version_id
    )

    versions = manager.list(
        "toy"
    )

    assert len(versions) == 2

    db.close()


def test_verify_detects_dataset_mutation(
    tmp_path: Path,
) -> None:

    dataset = tmp_path / "dataset"

    dataset.mkdir()

    file = dataset / "train.csv"

    file.write_text(
        "x\n1\n2\n",
        encoding="utf-8",
    )

    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    manager = DatasetVersionManager(db)

    version = manager.register(
        dataset_id="toy",
        path=dataset,
        source="local://toy",
    )

    manager.verify(
        version.version_id,
        dataset,
    )

    file.write_text(
        "x\n1\n999\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DatasetIntegrityError
    ):
        manager.verify(
            version.version_id,
            dataset,
        )

    db.close()


def test_same_contents_in_different_directory_have_same_version(
    tmp_path: Path,
) -> None:

    dataset_a = (
        tmp_path / "dataset_a"
    )

    dataset_b = (
        tmp_path / "dataset_b"
    )

    dataset_a.mkdir()
    dataset_b.mkdir()

    contents = "x\n1\n2\n"

    (
        dataset_a / "train.csv"
    ).write_text(
        contents,
        encoding="utf-8",
    )

    (
        dataset_b / "train.csv"
    ).write_text(
        contents,
        encoding="utf-8",
    )

    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    manager = DatasetVersionManager(db)

    first = manager.register(
        dataset_id="toy",
        path=dataset_a,
        source="local://a",
    )

    second = manager.register(
        dataset_id="toy",
        path=dataset_b,
        source="local://b",
    )

    assert (
        first.version_id
        == second.version_id
    )

    db.close()


def test_empty_directory_is_rejected(
    tmp_path: Path,
) -> None:

    dataset = tmp_path / "empty"

    dataset.mkdir()

    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    manager = DatasetVersionManager(db)

    with pytest.raises(ValueError):
        manager.register(
            dataset_id="empty",
            path=dataset,
            source="local://empty",
        )

    db.close()


def test_symlink_is_rejected(
    tmp_path: Path,
) -> None:

    actual = tmp_path / "actual.txt"

    actual.write_text(
        "data",
        encoding="utf-8",
    )

    link = (
        tmp_path / "link.txt"
    )

    link.symlink_to(
        actual
    )

    db = ResultsDatabase(
        tmp_path / "results.sqlite"
    )

    manager = DatasetVersionManager(db)

    with pytest.raises(ValueError):
        manager.register(
            dataset_id="symlink-test",
            path=link,
            source="local://symlink-test",
        )

    db.close()