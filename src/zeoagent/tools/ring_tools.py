"""Public ring analysis tools for zeolite structures."""

from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from itertools import repeat
from pathlib import Path
from typing import Dict, List, Optional

from pydantic import BaseModel

from zeoagent.config import get_settings


class RingTypeRequest(BaseModel):
    cif_dir: str
    max_ring: int = 12
    num_workers: int = 10
    validation: Optional[str] = "sastre"
    preferred_validation: Optional[str] = None
    framework: Optional[str] = None
    output_path: Optional[str] = None


class RingTypeEntry(BaseModel):
    cif_path: str
    ring_type_counts: Dict[int, int]
    ring_types: List[int]
    max_ring: int
    error: Optional[str] = None


class RingTypeResult(BaseModel):
    entries: List[RingTypeEntry]


class RingSizeRequest(BaseModel):
    cif_dir: str
    target_ring_size: int = 8
    bond_threshold: float = 3.5
    subtract_radius: float = 2.7
    validation: Optional[str] = "sastre"


class RingSizeEntry(BaseModel):
    cif_path: str
    ring_sizes: List[Dict[str, float]]
    min_dmin: Optional[float]
    max_dmin: Optional[float]
    min_dmax: Optional[float]
    max_dmax: Optional[float]
    error: Optional[str] = None


class RingSizeResult(BaseModel):
    entries: List[RingSizeEntry]


def _load_atoms(cif_path: Path):
    from ase.io import read

    return read(str(cif_path))


def _group_equivalent_oxygens(atoms) -> List[List[int]]:
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    import numpy as np

    structure = AseAtomsAdaptor.get_structure(atoms)
    sga = SpacegroupAnalyzer(structure)
    dataset = sga.get_symmetry_dataset()
    eq_atoms = np.array(dataset.equivalent_atoms)

    o_indices = [i for i, atom in enumerate(atoms) if atom.symbol == "O"]

    o_groups: Dict[int, List[int]] = defaultdict(list)
    for i in o_indices:
        o_groups[int(eq_atoms[i])].append(i)
    return list(o_groups.values())


def _ensure_zse_available() -> None:
    settings = get_settings()
    zse_path = settings.zse_path
    if zse_path:
        resolved = settings.resolve_path(zse_path)
        if str(resolved) not in sys.path:
            sys.path.insert(0, str(resolved))
    try:
        import zse  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "The zse package is required for ring analysis. Set ZSE_PATH or install zse in your environment."
        ) from exc


def _ensure_cif_temp_dir() -> None:
    _ensure_zse_available()
    from zse import rings as zse_rings

    temp_dir = Path(zse_rings.__file__).resolve().parent / ".temp_files"
    temp_dir.mkdir(parents=True, exist_ok=True)


def _normalize_validation(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = value.strip().lower()
    if normalized in {"", "none", "null", "false", "off"}:
        return None
    if normalized in {"auto", "automatic"}:
        return "sastre"
    return normalized


def _ring_debug_enabled() -> bool:
    value = (os.getenv("RING_DEBUG") or "").strip().lower()
    return value in {"1", "true", "yes", "on", "debug"}


def _ring_debug(message: str) -> None:
    if _ring_debug_enabled():
        print(f"[ring_debug] {message}")


def _unique_rings_from_tsites(
    cif_path: Path,
    atoms,
    max_ring: int,
    validation: Optional[str],
) -> Dict[int, int]:
    _ensure_zse_available()
    from zse import cif_tools, rings

    _ensure_cif_temp_dir()
    _, _, tinds = cif_tools.get_tsites(str(cif_path))
    if not tinds:
        raise ValueError(f"No T-sites found in {cif_path}")

    ring_list, _, _, _ = rings.get_unique_rings(
        atoms,
        tinds,
        validation=validation,
        max_ring=max_ring,
    )
    counts: Counter[int] = Counter(ring_list)
    return dict(sorted(counts.items()))


def _unique_rings_from_osites(
    atoms,
    max_ring: int,
    validation: Optional[str],
) -> Dict[int, int]:
    _ensure_zse_available()
    from zse.rings import get_rings
    from zse.ring_utilities import remove_dups, remove_geometric_dups

    o_groups = _group_equivalent_oxygens(atoms)
    if not o_groups:
        raise ValueError("No O-sites found in structure")

    paths: List[List[int]] = []
    last_atoms = None
    for group in o_groups:
        index = min(group)
        _, ring_paths, _, large_atoms = get_rings(atoms, index, validation=validation, max_ring=max_ring)
        if ring_paths:
            paths.extend(ring_paths)
            last_atoms = large_atoms

    if not paths:
        return {}

    paths = remove_dups(paths)
    if last_atoms is None:
        last_atoms = atoms
    paths = remove_geometric_dups(last_atoms, paths)
    ring_list = [int(len(path) / 2) for path in paths]
    counts: Counter[int] = Counter(ring_list)
    return dict(sorted(counts.items()))


def _collect_unique_ring_counts(
    cif_path: Path,
    atoms,
    max_ring: int,
    validation: Optional[str],
) -> Dict[int, int]:
    method = _normalize_validation(validation) or "sastre"
    try:
        counts = _unique_rings_from_tsites(cif_path, atoms, max_ring=max_ring, validation=method)
        _ring_debug(f"{cif_path.name}: method={method!r} path=tsites ring_types={sorted(counts.keys())}")
        return counts
    except Exception as exc:
        _ring_debug(f"{cif_path.name}: method={method!r} path=tsites failed: {exc}; fallback=osites")
        counts = _unique_rings_from_osites(atoms, max_ring=max_ring, validation=method)
        _ring_debug(f"{cif_path.name}: method={method!r} path=osites ring_types={sorted(counts.keys())}")
        return counts


def _write_json_output(result: BaseModel, output_path: Optional[str]) -> None:
    if not output_path:
        return
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result.model_dump(), indent=2), encoding="utf-8")


def get_max_ring_for_cif(
    cif_path: str,
    max_ring: int = 12,
    validation: Optional[str] = "sastre",
    framework: Optional[str] = None,
    preferred: Optional[str] = None,
) -> int:
    del framework, preferred
    path = Path(cif_path)
    if not path.exists():
        raise FileNotFoundError(f"CIF not found: {path}")
    atoms = _load_atoms(path)
    ring_counts = _collect_unique_ring_counts(path, atoms, max_ring, validation)
    ring_types = sorted(ring_counts.keys())
    return max(ring_types) if ring_types else 0


def _compute_ring_type_entry(
    cif_path: str,
    max_ring: int,
    validation: Optional[str],
) -> RingTypeEntry:
    path = Path(cif_path)
    try:
        atoms = _load_atoms(path)
        ring_counts = _collect_unique_ring_counts(path, atoms, max_ring, validation)
        ring_types = sorted(ring_counts.keys())
        max_ring_value = max(ring_types) if ring_types else 0
        return RingTypeEntry(
            cif_path=str(path),
            ring_type_counts=ring_counts,
            ring_types=ring_types,
            max_ring=max_ring_value,
        )
    except Exception as exc:
        return RingTypeEntry(
            cif_path=str(path),
            ring_type_counts={},
            ring_types=[],
            max_ring=0,
            error=str(exc),
        )


def run_ring_type_calculator(request: RingTypeRequest) -> RingTypeResult:
    """Return ring counts using an explicit validation method."""
    cif_dir = Path(request.cif_dir)
    if not cif_dir.exists():
        raise FileNotFoundError(f"CIF dir not found: {cif_dir}")

    validation = _normalize_validation(request.validation)
    if cif_dir.is_file():
        cif_paths = [cif_dir]
    else:
        cif_paths = sorted(cif_dir.glob("*.cif"))

    if not cif_paths:
        result = RingTypeResult(entries=[])
        _write_json_output(result, request.output_path)
        return result

    max_workers = max(1, int(request.num_workers))
    if max_workers > 1 and len(cif_paths) > 1:
        max_workers = min(max_workers, len(cif_paths))
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            entries = list(
                executor.map(
                    _compute_ring_type_entry,
                    (str(path) for path in cif_paths),
                    repeat(request.max_ring),
                    repeat(validation),
                )
            )
    else:
        entries = [
            _compute_ring_type_entry(str(path), request.max_ring, validation)
            for path in cif_paths
        ]

    result = RingTypeResult(entries=entries)
    _write_json_output(result, request.output_path)
    return result


def _get_ring_diameter_metrics(path, atoms, bond_threshold: float, subtract_radius: float):
    import networkx as nx

    o_indices = [idx for idx in path if atoms[idx].symbol == "O"]
    if len(o_indices) != 8:
        raise ValueError(f"Ring does not contain 8 O atoms (got {len(o_indices)})")

    graph = nx.Graph()
    graph.add_nodes_from(o_indices)
    for i in range(len(o_indices)):
        for j in range(i + 1, len(o_indices)):
            distance = atoms.get_distance(o_indices[i], o_indices[j], mic=True)
            if distance < bond_threshold:
                graph.add_edge(o_indices[i], o_indices[j])

    try:
        cycle = nx.find_cycle(graph)
    except nx.NetworkXNoCycle as exc:
        raise RuntimeError("No closed 8-ring cycle detected") from exc

    ordered_o_indices: List[int] = []
    visited = set()
    for u, v in cycle:
        if u not in visited:
            ordered_o_indices.append(u)
            visited.add(u)
        if v not in visited:
            ordered_o_indices.append(v)
            visited.add(v)
        if len(ordered_o_indices) == 8:
            break

    if len(ordered_o_indices) != 8:
        raise RuntimeError("Incomplete 8-ring ordering detected")

    distances: List[float] = []
    for i in range(4):
        idx1 = ordered_o_indices[i]
        idx2 = ordered_o_indices[i + 4]
        distances.append(atoms.get_distance(idx1, idx2, mic=True))

    dmax = max(distances) - subtract_radius
    dmin = min(distances) - subtract_radius
    return dmax, dmin


def run_ring_size_calculator(request: RingSizeRequest) -> RingSizeResult:
    """Compute ring aperture metrics using a user-selected validation method."""
    cif_dir = Path(request.cif_dir)
    if not cif_dir.exists():
        raise FileNotFoundError(f"CIF dir not found: {cif_dir}")

    validation = _normalize_validation(request.validation)
    entries: List[RingSizeEntry] = []
    for cif_path in sorted(cif_dir.glob("*.cif")):
        try:
            atoms = _load_atoms(cif_path)
            unique_paths = []
            seen = set()
            for group in _group_equivalent_oxygens(atoms):
                index = min(group)
                _ensure_zse_available()
                from zse.rings import get_rings

                _, paths, _, ring_atoms = get_rings(
                    atoms,
                    index,
                    validation=validation,
                    max_ring=request.target_ring_size,
                )
                for path in paths:
                    if len(path) != request.target_ring_size * 2:
                        continue
                    key = tuple(sorted(path))
                    if key in seen:
                        continue
                    seen.add(key)
                    unique_paths.append((path, ring_atoms))

            ring_sizes: List[Dict[str, float]] = []
            for path, ring_atoms in unique_paths:
                dmax, dmin = _get_ring_diameter_metrics(
                    path,
                    ring_atoms,
                    bond_threshold=request.bond_threshold,
                    subtract_radius=request.subtract_radius,
                )
                ring_sizes.append({"dmax": round(dmax, 2), "dmin": round(dmin, 2)})

            dmax_values = [item["dmax"] for item in ring_sizes]
            dmin_values = [item["dmin"] for item in ring_sizes]
            entries.append(
                RingSizeEntry(
                    cif_path=str(cif_path),
                    ring_sizes=ring_sizes,
                    min_dmin=min(dmin_values) if dmin_values else None,
                    max_dmin=max(dmin_values) if dmin_values else None,
                    min_dmax=min(dmax_values) if dmax_values else None,
                    max_dmax=max(dmax_values) if dmax_values else None,
                )
            )
        except Exception as exc:
            entries.append(
                RingSizeEntry(
                    cif_path=str(cif_path),
                    ring_sizes=[],
                    min_dmin=None,
                    max_dmin=None,
                    min_dmax=None,
                    max_dmax=None,
                    error=str(exc),
                )
            )
    return RingSizeResult(entries=entries)


__all__ = [
    "RingTypeRequest",
    "RingTypeEntry",
    "RingTypeResult",
    "RingSizeRequest",
    "RingSizeEntry",
    "RingSizeResult",
    "get_max_ring_for_cif",
    "run_ring_type_calculator",
    "run_ring_size_calculator",
]
