"""Utilities for resolving zeolite CIF file paths."""

from pathlib import Path
from typing import Dict

from zeoagent.config import get_settings

# Simple alias map; keys are normalized (upper, no spaces/hyphens).
ALIASES: Dict[str, str] = {
    "SAPO34": "CHA",
    "SAPO-34": "CHA",  # allow hyphenated form before normalization
}


def _normalize_framework_name(name: str) -> str:
    """Normalize framework name for lookup (case-insensitive, strip ext/spaces/hyphens)."""
    stem = Path(name).stem  # drop any extension if provided
    canonical = stem.strip().upper().replace(" ", "").replace("-", "")
    return ALIASES.get(canonical, canonical)


def resolve_cif_path(framework_name: str) -> Path:
    """
    Resolve a framework name to a CIF file path within the configured CIF directory.

    - Case-insensitive.
    - Supports simple aliases (e.g., SAPO-34 -> CHA).
    - Raises FileNotFoundError with details if not present.
    """
    if not framework_name or not framework_name.strip():
        raise ValueError("framework_name is required")

    settings = get_settings()
    cif_root = settings.resolve_path(settings.cif_dir)
    canonical = _normalize_framework_name(framework_name)

    candidates = [
        cif_root / f"{canonical}.cif",
        cif_root / f"{canonical.lower()}.cif",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(
        f"CIF file for framework '{framework_name}' not found in {cif_root}. Tried: {tried}"
    )


__all__ = ["resolve_cif_path"]
