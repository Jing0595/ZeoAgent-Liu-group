#!/usr/bin/env python3
"""Public manuscript-facing entry point for the point-cloud constrained workflow."""

from __future__ import annotations

from pathlib import Path
import runpy


if __name__ == "__main__":
    target = Path(__file__).with_name("generate_frameworks_from_contour.py")
    runpy.run_path(str(target), run_name="__main__")
