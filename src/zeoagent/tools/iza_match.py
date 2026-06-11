"""IZA match tool for generated zeolite CIFs."""

from __future__ import annotations

import pickle
import json
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel, Field

from zeoagent.config import get_settings


class IzaMatchRequest(BaseModel):
    cif_dir: str
    iza_dir: Optional[str] = None
    cache_path: Optional[str] = None
    output_path: Optional[str] = None
    rebuild_cache: bool = False
    tt_min: float = Field(default=2.9, description="Minimum Si-Si distance for topology graph.")
    tt_max: float = Field(default=4.2, description="Maximum Si-Si distance for topology graph.")


class IzaMatchEntry(BaseModel):
    cif_path: str
    iza_code: Optional[str] = None
    status: str
    error: Optional[str] = None


class IzaMatchResult(BaseModel):
    entries: List[IzaMatchEntry]


def _resolve_iza_dir(iza_dir: Optional[str]) -> Path:
    settings = get_settings()
    if iza_dir:
        path = Path(iza_dir)
        return settings.resolve_path(path)
    if getattr(settings, "iza_cif_dir", None) is not None:
        return settings.resolve_path(settings.iza_cif_dir)
    raise ValueError("IZA CIF directory is not configured")


def _cache_path_for(iza_dir: Path, cache_path: Optional[str]) -> Path:
    if cache_path:
        return Path(cache_path)
    return iza_dir / ".iza_graphs_cache.pkl"


def _is_cache_valid(cache_path: Path, iza_dir: Path) -> bool:
    if not cache_path.exists():
        return False
    cache_mtime = cache_path.stat().st_mtime
    for cif_file in iza_dir.glob("*.cif"):
        if cif_file.stat().st_mtime > cache_mtime:
            return False
    return True


def _load_cache(cache_path: Path) -> Optional[Dict[str, object]]:
    try:
        with cache_path.open("rb") as fh:
            return pickle.load(fh)
    except Exception:
        return None


def _save_cache(cache_path: Path, iza_graphs: Dict[str, object]) -> None:
    try:
        with cache_path.open("wb") as fh:
            pickle.dump(iza_graphs, fh)
    except Exception:
        return


def _write_json_output(result: BaseModel, output_path: Optional[str]) -> None:
    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.model_dump(), indent=2), encoding="utf-8")


def _build_tt_graph(struct, tt_min: float, tt_max: float):
    import networkx as nx
    import numpy as np

    si_indices = [i for i, site in enumerate(struct) if site.specie.symbol == "Si"]
    frac_coords = struct.frac_coords[si_indices]
    lattice = struct.lattice

    def min_image_dist(p, q):
        dfrac = q - p
        dfrac -= np.round(dfrac)
        vec = dfrac @ lattice.matrix
        return np.linalg.norm(vec)

    graph = nx.Graph()
    for idx in range(len(si_indices)):
        graph.add_node(idx)
    for i in range(len(si_indices)):
        for j in range(i + 1, len(si_indices)):
            dist = min_image_dist(frac_coords[i], frac_coords[j])
            if tt_min < dist < tt_max:
                graph.add_edge(i, j)
    return graph


def _build_iza_database(iza_dir: Path, tt_min: float, tt_max: float) -> Dict[str, object]:
    from pymatgen.core import Structure

    iza_graphs: Dict[str, object] = {}
    for cif_path in sorted(iza_dir.glob("*.cif")):
        iza_code = cif_path.stem
        try:
            struct = Structure.from_file(cif_path)
            iza_graphs[iza_code] = _build_tt_graph(struct, tt_min, tt_max)
        except Exception:
            continue
    if not iza_graphs:
        raise RuntimeError(f"No IZA CIFs loaded from {iza_dir}")
    return iza_graphs


def _load_iza_database(
    iza_dir: Path,
    cache_path: Path,
    rebuild_cache: bool,
    tt_min: float,
    tt_max: float,
) -> Dict[str, object]:
    if not rebuild_cache and _is_cache_valid(cache_path, iza_dir):
        cached = _load_cache(cache_path)
        if cached:
            return cached
    iza_graphs = _build_iza_database(iza_dir, tt_min, tt_max)
    _save_cache(cache_path, iza_graphs)
    return iza_graphs


def run_iza_match(request: IzaMatchRequest) -> IzaMatchResult:
    """Match CIFs against the IZA database and report per-CIF status."""
    cif_dir = Path(request.cif_dir)
    if not cif_dir.exists():
        raise FileNotFoundError(f"CIF dir not found: {cif_dir}")

    iza_dir = _resolve_iza_dir(request.iza_dir)
    if not iza_dir.exists():
        raise FileNotFoundError(f"IZA CIF dir not found: {iza_dir}")

    cache_path = _cache_path_for(iza_dir, request.cache_path)
    iza_graphs = _load_iza_database(iza_dir, cache_path, request.rebuild_cache, request.tt_min, request.tt_max)

    import networkx as nx
    from pymatgen.core import Structure

    entries: List[IzaMatchEntry] = []
    for cif_path in sorted(cif_dir.glob("*.cif")):
        try:
            struct = Structure.from_file(cif_path)
            test_graph = _build_tt_graph(struct, request.tt_min, request.tt_max)
            matched_code = None
            test_nodes = test_graph.number_of_nodes()
            test_edges = test_graph.number_of_edges()
            for iza_code, iza_graph in iza_graphs.items():
                if (
                    test_nodes != iza_graph.number_of_nodes()
                    or test_edges != iza_graph.number_of_edges()
                ):
                    continue
                if nx.is_isomorphic(test_graph, iza_graph):
                    matched_code = iza_code
                    break
            if matched_code:
                entries.append(IzaMatchEntry(cif_path=str(cif_path), iza_code=matched_code, status="matched"))
            else:
                entries.append(IzaMatchEntry(cif_path=str(cif_path), status="new"))
        except Exception as exc:
            entries.append(IzaMatchEntry(cif_path=str(cif_path), status="error", error=str(exc)))

    result = IzaMatchResult(entries=entries)
    _write_json_output(result, request.output_path)
    return result


__all__ = ["IzaMatchRequest", "IzaMatchEntry", "IzaMatchResult", "run_iza_match"]
