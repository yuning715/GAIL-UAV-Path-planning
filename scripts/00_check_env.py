from __future__ import annotations

import argparse
import importlib.util
import platform
from pathlib import Path

from _common import REPO_ROOT, add_common_args, record_command


def main() -> None:
    parser = add_common_args(argparse.ArgumentParser())
    parser.parse_args()
    record_command()
    required = ["numpy", "pandas", "matplotlib", "yaml", "torch", "pytest"]
    print(f"Repository: {REPO_ROOT}")
    print(f"Python: {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    for module in required:
        print(f"{module}: {'OK' if importlib.util.find_spec(module) else 'MISSING'}")
    for path in ["data/expert", "data/processed", "outputs/logs", "outputs/figures", "outputs/tables"]:
        p = REPO_ROOT / path
        p.mkdir(parents=True, exist_ok=True)
        print(f"{path}: {p}")


if __name__ == "__main__":
    main()
