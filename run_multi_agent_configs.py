#!/usr/bin/env python3
"""Batch runner for the multi-agent simulation."""

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
SIM_SCRIPT = REPO_ROOT / "multi-agent.py"


def collect_configs(paths, pattern):
    """Yield config files from a mix of files and directories."""
    for raw_path in paths:
        path = Path(raw_path).expanduser()
        if path.is_dir():
            yield from sorted(path.glob(pattern))
        elif path.is_file():
            yield path
        else:
            raise FileNotFoundError(f"Path not found: {path}")


def run_config(config_path):
    """Run the simulation for a single config file."""
    abs_config = config_path.resolve()
    print(f"\n>>> Running config: {abs_config}")
    cmd = [sys.executable, str(SIM_SCRIPT), "--config", str(abs_config)]
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run the multi-agent simulation for one or many config files."
    )
    parser.add_argument(
        "paths",
        nargs="+",
        help="Config file(s) or directories containing config files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.json",
        help="Glob used when paths include directories (default: *.json).",
    )
    args = parser.parse_args()

    configs = list(collect_configs(args.paths, args.pattern))
    if not configs:
        print("No config files found.", file=sys.stderr)
        sys.exit(1)

    for config_path in configs:
        run_config(config_path)


if __name__ == "__main__":
    main()
