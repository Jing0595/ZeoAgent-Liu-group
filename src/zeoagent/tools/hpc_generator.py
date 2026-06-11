"""Manual HPC integration contract for the open-source release."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel


class HPCJobRequest(BaseModel):
    framework: str
    threshold: float = 2.2
    excl: float = 0.81
    scale_scan: str = "1.0:1.4:0.1"
    tden_min: float = 10.0
    num_unique_list: str = "2 3"
    num_seeds: int = 15
    allow_reuse: bool = True
    retry_on_empty: bool = False
    retry_num_seeds: int = 20
    retry_threshold_step: float = 0.2
    poll_interval_seconds: int = 10
    max_poll_attempts: int = 360
    candidate_dir: Optional[str] = None


class HPCJobResult(BaseModel):
    job_id: Optional[str]
    framework: str
    remote_run_dir: str
    local_output_dir: str
    fetched_paths: List[str]
    iza_match_status: Dict[str, str] = {}
    reused_run: bool = False
    retried_on_empty: bool = False
    retry_count: int = 0
    failed_after_retries: bool = False
    manual_action_required: bool = False
    instructions: Optional[str] = None


def _collect_candidate_paths(candidate_dir: Path) -> List[str]:
    return [str(path) for path in sorted(candidate_dir.glob("*.cif")) if path.is_file()]


def run_hpc_generation(request: HPCJobRequest) -> HPCJobResult:
    """Consume user-prepared candidate CIFs or return manual deployment instructions."""
    if request.candidate_dir:
        candidate_dir = Path(request.candidate_dir)
        if candidate_dir.exists():
            fetched = _collect_candidate_paths(candidate_dir)
            if fetched:
                return HPCJobResult(
                    job_id=None,
                    framework=request.framework,
                    remote_run_dir="",
                    local_output_dir=str(candidate_dir),
                    fetched_paths=fetched,
                )

    candidate_dir = Path("data/generated_candidates") / request.framework.upper()
    instructions = (
        "The open-source ZeoAgent release does not include the proprietary remote generation "
        "workflow. Deploy the public point-cloud generation algorithm on your own HPC system, "
        f"run it for framework {request.framework.upper()}, and return the generated candidate "
        f"CIF files to {candidate_dir}. Then rerun this step with candidate_dir pointing to the "
        "local directory that contains those CIF files so ZeoAgent can continue downstream "
        "screening with ring analysis, Zeo++ metrics, and IZA matching. See docs/hpc_adaptation.md."
    )
    raise RuntimeError(instructions)


__all__ = ["HPCJobRequest", "HPCJobResult", "run_hpc_generation"]
