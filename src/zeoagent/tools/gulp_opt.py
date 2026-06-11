"""Manual GULP optimization step for the open-source release."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field

from zeoagent.tools.cif_resolver import resolve_cif_path


class GulpOptimizationRequest(BaseModel):
    framework: Optional[str] = None
    cif_path: Optional[str] = None
    output_dir: Optional[str] = None
    output_cif: Optional[str] = None
    stepmx: float = Field(default=0.1, ge=0.0)
    run_label: Optional[str] = None


class GulpOptimizationResult(BaseModel):
    input_cif_path: str
    optimized_cif_path: Optional[str] = None
    run_dir: Optional[str] = None
    instructions: str


def run_gulp_optimization(request: GulpOptimizationRequest) -> GulpOptimizationResult:
    """Return instructions instead of exposing the internal GULP workflow."""
    if request.cif_path:
        input_cif = Path(request.cif_path)
    elif request.framework:
        input_cif = resolve_cif_path(request.framework)
    else:
        raise ValueError("Either framework or cif_path must be provided for GULP optimization")

    expected_output = request.output_cif or "optimized_reference.cif"
    instructions = (
        "This open-source release does not bundle the internal GULP force-field setup or "
        "input script templates. Optimize the reference CIF with your own GULP workflow, "
        f"using {input_cif} as input, and save the optimized structure as {expected_output}. "
        "After the optimized CIF is available locally, point ZeoAgent to that file for "
        "downstream ring, pore, and novelty analysis."
    )
    raise RuntimeError(instructions)


__all__ = [
    "GulpOptimizationRequest",
    "GulpOptimizationResult",
    "run_gulp_optimization",
]
