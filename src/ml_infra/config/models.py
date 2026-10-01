from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


# inheriting from BaseModel, RunConfig automatically inherits all of Pydantic's built-in
class RunConfig(BaseModel):
    """
    Configuration describing one reproducible ML experiment run.
    """

    # Catches typos immediately. By default, Pydantic ignores extra fields.
    # Setting extra="forbid" means if someone writes seeeed: 42 or lr: 1e-4 at the root level of their YAML instead of inside parameters.
    # Pydantic will raise a ValidationError and stop execution rather than silently dropping the parameter.
    model_config = ConfigDict(extra="forbid")

    # both fields are mandatory (no default value) and must contain at least 1 character (min_length=1).
    # This prevents unnamed runs from overwriting logs or creating directory collisions
    project: str = Field(min_length=1)
    run_name: str = Field(min_length=1)

    # Defaults to 42, but enforces ge=0 (greater than or equal to 0). Negative seeds or non-integer inputs are rejected.
    seed: int = Field(default=42, ge=0)

    # File management. It automatically converts incoming path strings (e.g., "./runs/baseline") into standard Python pathlib.Path objects
    output_dir: Path = Field(default=Path("runs"))

    # Hyperparameter container
    parameters: dict[str, Any] = Field(default_factory=dict)

    # Queryable metadata. Stores string-to-string key-value pairs (e.g., {"optimizer": "adamw", "hardware": "a100"})
    # used later by your tracker or results database to filter and compare experiments.
    tags: dict[str, str] = Field(default_factory=dict)