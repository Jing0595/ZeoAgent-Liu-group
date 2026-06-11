"""Public configuration for the open-source ZeoAgent release."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or ``.env``."""

    qwen_api_key: Optional[str] = Field(default=None, alias="QWEN_API_KEY")
    gpt5_api_key: Optional[str] = Field(default=None, alias="GPT5_API_KEY")

    data_root: Path = Path("./data")
    cif_dir: Path = Path("data/cif_files")
    iza_cif_dir: Path = Path("data/iza_cif")
    zeopp_bin: Path = Path("tools/zeopp/network")
    zse_path: Optional[Path] = Field(default=None, alias="ZSE_PATH")

    separation_corpus_dir: Path = Path("data/separation_corpus")
    molecular_diameter_path: Path = Path("data/reference/molecular_diameters.json")

    generation_model: str = Field(default="gpt-5.2", alias="GENERATION_MODEL")
    generation_base_url: Optional[str] = Field(default=None, alias="GENERATION_BASE_URL")
    debug_trace_raw: bool = Field(default=False, alias="ZEOAGENT_DEBUG_TRACE_RAW")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    def resolve_path(self, path: Path) -> Path:
        """Resolve a project-relative path from the repository root."""
        if path.is_absolute():
            return path
        project_root = Path(__file__).resolve().parents[2]
        return project_root / path


@lru_cache()
def get_settings() -> Settings:
    """Return a cached settings object."""
    settings = Settings()
    if settings.zse_path and "ZSE_PATH" not in os.environ:
        os.environ["ZSE_PATH"] = str(settings.resolve_path(settings.zse_path))
    return settings
