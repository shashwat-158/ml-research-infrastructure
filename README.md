# ml-research-infrastructure

Shared research tooling built for my [AI Research Ecosystem](https://github.com/shashwat-158) collection — providing experiment tracking, config-driven runs, and a common evaluation framework across multiple ML agents.

**Status:** in progress

## Planned components
- [x] Experiment tracker (config + seed → result, queryable)
- [x] Config system (runs driven by files, not hardcoded values)
- [ ] Dataset versioning
- [x] Checkpoint manager
- [ ] Shared evaluation framework
- [ ] Unified logging
- [ ] Benchmark runner
- [ ] GPU monitoring
- [x] Results database

## Usage

Other projects can consume the infrastructure as a Python package:

```python
from ml_infra.config import load_config
from ml_infra.results_db import ResultsDatabase
from ml_infra.tracker import ExperimentTracker

config = load_config("configs/experiment.yaml")

db = ResultsDatabase("runs/results.sqlite")
tracker = ExperimentTracker(db)

with tracker.run(config) as run:
    run.log_metric("loss", 1.23, step=100)
```

Each run records its configuration snapshot, configuration hash,
random seed, Git commit, dataset version, lifecycle status, metrics,
parameters, tags, and registered artifacts.

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

## Used by
- 