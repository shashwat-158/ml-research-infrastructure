# Standard library module used to build user-friendly command-line interfaces
import argparse
import json

from .loader import config_hash, load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load and validate an ML experiment configuration."
    )

    # When running the script, the user must supply the path to a YAML file (e.g., python script.py configs/baseline.yaml).
    parser.add_argument(
        "config",
        help="Path to a YAML configuration file.",
    )

    args = parser.parse_args()

    config = load_config(args.config)

    print("Configuration loaded successfully.")
    print()
    print("Resolved configuration:")
    print(
        json.dumps(
            config.model_dump(mode="json"),
            indent=2,
        )
    )

    print()
    print(f"Config hash: {config_hash(config)}")


if __name__ == "__main__":
    main()