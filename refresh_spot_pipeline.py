#!/usr/bin/env python3
"""One-click refresh of the spot pipeline (data -> profiles -> weights -> models)."""

from __future__ import annotations

import importlib
import sys
from typing import Sequence

STEPS: Sequence[str] = (
    "count_downward_impulses",
    "extract_downward_profiles",
    "extract_upward_profiles",
    "compute_downtrend_weights",
    "compute_uptrend_weights",
    "train_spot_long_models",
)


def run_step(module_name: str) -> None:
    module = importlib.import_module(module_name)
    if not hasattr(module, "main"):
        raise RuntimeError(f"Module {module_name} does not expose main()")
    print(f"\n=== Running {module_name} ===")
    module.main()  # type: ignore[attr-defined]


def main() -> None:
    for step in STEPS:
        run_step(step)
    print("\nPipeline finished successfully.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - top-level safety net
        print(f"Pipeline failed: {exc}", file=sys.stderr)
        sys.exit(1)
