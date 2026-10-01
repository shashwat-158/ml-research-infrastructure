# ml-research-infrastructure

Shared research tooling built for my [AI Research Ecosystem](https://github.com/shashwat-158) collection — providing experiment tracking, config-driven runs, and a common evaluation framework across multiple ML agents.

**Status:** in progress

## Components
- [x] Experiment tracker (config + seed → result, queryable)
- [x] Config system (runs driven by files, not hardcoded values)
- [x] Dataset versioning
- [x] Checkpoint manager
- [x] Shared evaluation framework
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

# Evaluation
from ml_infra.evaluation import (
    EvaluationCase,
    EvaluationRunner,
    EvaluationSuite,
    ExactMatchEvaluator,
)

suite = EvaluationSuite(
    name="basic_reasoning",
    cases=(
        EvaluationCase(
            case_id="case_001",
            input="17 * 19",
            expected="323",
        ),
    ),
)

runner = EvaluationRunner(db)

report = runner.run(
    suite,
    model,
    [ExactMatchEvaluator()],
)
```
Consuming repositories depend on these public interfaces rather than
SQLite tables or internal implementation modules.

## Reproducibility Model

```text
             Experiment
                  │
      ┌───────────┼────────────┐
      ▼           ▼            ▼
    Config       Code        Dataset
      │           │            │
 config_hash  git_commit   version_id
      │           │            │
      └───────────┼────────────┘
                  ▼
             Run identity
                  │
       ┌──────────┼──────────┐
       ▼          ▼          ▼
 Checkpoints   Metrics   Evaluations
    │                        │
 SHA-256              ┌─────┴──────┐
                      ▼            ▼
                  case results   metrics
                      │
                  suite_hash
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

## Evaluation Framework

The shared evaluation framework separates system execution from grading.

An evaluation consists of:

```text
Evaluation Suite
    │
    ├── Evaluation Cases
    │
    ├── Evaluators
    │
    └── Evaluation Runner
            │
            ▼
       Case Results
            │
            ▼
       Aggregated Metrics
```
Each evaluation records:

- evaluation ID
- suite name and deterministic suite hash
- evaluator names
- optional associated experiment run
- random seed when applicable
- case-level results
- grader scores and pass/fail decisions
- explanations and structured details
- aggregate metrics
- evaluation lifecycle status

The framework is intentionally separate from benchmark orchestration:
the evaluation framework defines how one system is evaluated, while the
future benchmark runner will coordinate many evaluations across models,
suites, configurations, and seeds.

## Used by

Not yet integrated by another ecosystem repository.

## Current Milestone
Implemented:

- configuration validation
- deterministic configuration hashing
- experiment lifecycle tracking
- SQLite results database
- incremental database migrations
- metric and parameter logging
- tags
- artifact registration
- atomic checkpoint persistence
- SHA-256 checkpoint integrity verification
- latest/best checkpoint tracking
- content-addressed dataset versioning
- dataset integrity verification
- shared evaluation cases and suites
- deterministic evaluation suite hashing
- pluggable evaluator protocol
- exact-match and numeric-tolerance evaluators
- custom callable evaluators
- case-level evaluation results
- aggregate evaluation metrics
- evaluation lifecycle tracking
- evaluation-to-experiment linkage
- integration tests
