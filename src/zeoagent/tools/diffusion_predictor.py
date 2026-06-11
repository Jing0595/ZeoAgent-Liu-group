"""Public diffusion predictor interface for the open-source release."""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel


class DiffusionPrediction(BaseModel):
    """Structured diffusion prediction payload."""

    framework: str
    diffusion_coefficient: float
    model_name: Optional[str] = None
    descriptor_row_index: Optional[int] = None
    features_used: Optional[List[str]] = None
    raw_log_prediction: Optional[float] = None


def predict_ethene_diffusion(
    framework: str,
    temperature_K: Optional[float] = None,
    loading_per_uc: Optional[float] = None,
    unitcell_volume_a3: Optional[float] = None,
) -> DiffusionPrediction:
    """Raise a public-facing integration error for unavailable private assets."""
    del temperature_K, loading_per_uc, unitcell_volume_a3
    if not framework or not framework.strip():
        raise ValueError("framework is required")
    raise RuntimeError(
        "Diffusion prediction is unavailable in the default open-source release. "
        "ZeoAgent does not ship with the private pretrained model or descriptor dataset "
        "used in the internal project. To enable this tool, integrate your own predictor "
        "backend and descriptor source following docs/model_integration.md."
    )


__all__ = ["DiffusionPrediction", "predict_ethene_diffusion"]
