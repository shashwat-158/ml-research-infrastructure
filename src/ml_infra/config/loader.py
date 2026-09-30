import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from .models import RunConfig

# Defensive Ingestion, Rather than blindly trusting user-provided YAML files,
def load_config(path: str | Path) -> RunConfig:
    """
    Load a YAML configuration file and validate it as RunConfig.
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    if path.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            f"Unsupported config format: {path.suffix}. "
            "Expected .yaml or .yml"
        )

    # Parses the YAML content from the opened file object using PyYAML's safe_load function.
    # converting the YAML structure into a standard Python dictionary and assigning it to raw_config
    with path.open("r", encoding="utf-8") as file:
        raw_config: dict[str, Any] = yaml.safe_load(file)

    if raw_config is None:
        raise ValueError(f"Configuration file is empty: {path}")

    if not isinstance(raw_config, dict):
        raise ValueError("Top-level configuration must be a YAML mapping/object.")

    return RunConfig.model_validate(raw_config)


# Deterministic Experiment Identity
# naming runs manually (e.g., run_baseline_final_v2) leads to collisions and lost work.
def config_hash(config: RunConfig) -> str:
    """
    Create a deterministic hash representing the semantic experiment config.

    The output directory is intentionally excluded because changing where
    artifacts are stored should not create a different experiment definition.
    """

    # Converts the Pydantic model into JSON-serializable primitives
    payload = config.model_dump(mode="json")

    # Excluding output_dir ensures the hash stays identical regardless of where the code runs
    payload.pop("output_dir", None)

    canonical_json = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    # Encodes the canonical string to bytes
    return hashlib.sha256(
        canonical_json.encode("utf-8")
    ).hexdigest()[:12]