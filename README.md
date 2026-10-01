# ml-research-infrastructure

Shared research tooling built for my [AI Research Ecosystem](https://github.com/shashwat-158) collection — providing experiment tracking, config-driven runs, and a common evaluation framework across multiple ML agents.

**Status:** in progress

## Components
- [x] Experiment tracker (config + seed → result, queryable)
- [x] Config system (runs driven by files, not hardcoded values)
- [x] Dataset versioning
- [x] Checkpoint manager
- [ ] Shared evaluation framework
- [ ] Unified logging
- [ ] Benchmark runner
- [ ] GPU monitoring
- [x] Results database

## Reproducibility

Each experiment run records the information needed to connect a result
to the code, configuration, dataset, and generated artifacts.

Current run metadata includes:

- configuration snapshot and deterministic configuration hash
- random seed
- Git commit
- registered dataset version
- lifecycle status
- metric history
- parameters and tags
- registered artifacts

Dataset versions are content-addressed through a canonical file
manifest and SHA-256 identity.

Checkpoints additionally store size and SHA-256 integrity metadata and
support latest/best checkpoint selection and integrity-verified loading.

## Public Python API

The infrastructure is consumed through stable package-level interfaces:

```python
from ml_infra.config import load_config
from ml_infra.results_db import ResultsDatabase
from ml_infra.tracker import ExperimentTracker
from ml_infra.checkpoints import CheckpointManager

config = load_config(
    "configs/experiment.yaml"
)

db = ResultsDatabase(
    "runs/results.sqlite"
)

tracker = ExperimentTracker(db)

checkpoint_manager = CheckpointManager(
    db,
    root_dir="runs"
)

with tracker.run(config) as run:
    run.log_metric(
        "loss",
        1.23,
        step=100
    )
```
Consuming repositories depend on these public interfaces rather than
SQLite tables or internal implementation modules.

## Reproducibility Model

```text
             Experiment
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼
    Config       Code       Dataset
      │           │           │
 config_hash  git_commit  version_id
      │           │           │
      └───────────┼───────────┘
                  ▼
             Run identity
                  │
             ┌────┴────┐
             ▼         ▼
        Checkpoints  Metrics
             │
          SHA-256
```

## Checkpointing

Checkpoints are managed independently from model serialization.

The checkpoint manager provides:

- atomic file writes
- SHA-256 integrity verification
- checkpoint metadata
- latest checkpoint tracking
- best-checkpoint tracking by metric
- safe restore with integrity verification

Model-specific serialization is supplied by the consuming project.
For example, a PyTorch repository can provide `torch.save` / `torch.load`
through the checkpoint manager's writer/reader interface.

## Database migrations

The results database maintains an explicit schema version and applies
incremental migrations when the infrastructure is updated.

Existing experiment history is preserved when new infrastructure
components introduce additional tables.

## Dataset versioning

Datasets are registered as immutable content-based versions.

A dataset version is identified by:

```text
dataset contents
    ↓
file-level SHA-256 hashes
    ↓
canonical manifest
    ↓
SHA-256 version identity
```

## Used by

Not yet integrated by another ecosystem repository.
