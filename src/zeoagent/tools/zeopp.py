"""Core Zeo++ runner utilities (command builder, execution, parsing)."""

import glob
import re
import subprocess
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from zeoagent.config import get_settings
from zeoagent.tools.cif_resolver import resolve_cif_path


class ZeoppCalculationType(str, Enum):
    pore_diameters = "pore_diameters"
    surface_area = "surface_area"
    volume = "volume"
    channel_analysis = "channel_analysis"


class ZeoppRequest(BaseModel):
    framework: str
    cif_path: Optional[str] = None
    calculation_type: ZeoppCalculationType = ZeoppCalculationType.pore_diameters
    probe_radius: Optional[float] = None
    chan_radius: Optional[float] = None
    num_samples: Optional[int] = None
    extra_args: Optional[List[str]] = None
    channel_dedup_tolerance: float = 0.0001

    class Config:
        arbitrary_types_allowed = True


class PoreDiameterResult(BaseModel):
    largest_included_sphere: Optional[float] = None
    largest_free_sphere: Optional[float] = None
    largest_included_sphere_along_free_sphere_path: Optional[float] = None


class SurfaceAreaResult(BaseModel):
    asa_a2: Optional[float] = None
    asa_m2_per_cm3: Optional[float] = None
    asa_m2_per_g: Optional[float] = None
    nasa_a2: Optional[float] = None
    nasa_m2_per_cm3: Optional[float] = None
    nasa_m2_per_g: Optional[float] = None
    probe_radius: Optional[float] = None


class VolumeResult(BaseModel):
    av_a3: Optional[float] = None
    av_fraction: Optional[float] = None
    av_cm3_per_g: Optional[float] = None
    nav_a3: Optional[float] = None
    nav_fraction: Optional[float] = None
    nav_cm3_per_g: Optional[float] = None
    probe_radius: Optional[float] = None


class ChannelEntry(BaseModel):
    index: int
    dimensionality: Optional[int] = None
    di: Optional[float] = None
    df: Optional[float] = None
    dif: Optional[float] = None


class ChannelAnalysisResult(BaseModel):
    channel_count: int = 0
    dimensionalities: List[int] = Field(default_factory=list)
    channels: List[ChannelEntry] = Field(default_factory=list)
    probe_radius: Optional[float] = None


class ZeoppResult(BaseModel):
    framework: str
    calculation_type: ZeoppCalculationType
    pore_diameters: Optional[PoreDiameterResult] = None
    surface_area: Optional[SurfaceAreaResult] = None
    volume: Optional[VolumeResult] = None
    channel_analysis: Optional[ChannelAnalysisResult] = None
    raw_output: Optional[str] = None
    output_files: Optional[Dict[str, str]] = None


def build_zeopp_command(request: ZeoppRequest) -> Tuple[List[str], Dict[str, Path]]:
    """Build Zeo++ command and output file map based on the request."""
    settings = get_settings()
    cif_path = Path(request.cif_path) if request.cif_path else resolve_cif_path(request.framework)
    data_root = settings.resolve_path(settings.data_root)
    output_dir = data_root / "zeopp_runs" / Path(cif_path).stem
    output_dir.mkdir(parents=True, exist_ok=True)

    exe = settings.resolve_path(settings.zeopp_bin)
    base_cmd: List[str] = [str(exe), "-ha", "-stripatomnames"]
    outputs: Dict[str, Path] = {}

    calc_type = request.calculation_type
    extra_args = request.extra_args or []

    if calc_type == ZeoppCalculationType.pore_diameters:
        output_file = output_dir / f"{Path(cif_path).stem}.res"
        cmd = base_cmd + ["-res", str(output_file)] + extra_args + [str(cif_path)]
        outputs["res"] = output_file
        return cmd, outputs

    # surface area or volume require probe radius
    if request.probe_radius is None:
        raise ValueError("probe_radius is required for surface area and volume calculations")

    chan_radius = request.chan_radius if request.chan_radius is not None else request.probe_radius

    if calc_type == ZeoppCalculationType.surface_area:
        num_samples = request.num_samples if request.num_samples is not None else 2000
        output_file = output_dir / f"{Path(cif_path).stem}.sa"
        cmd = (
            base_cmd
            + ["-sa", str(chan_radius), str(request.probe_radius), str(num_samples)]
            + extra_args
            + [str(output_file), str(cif_path)]
        )
        outputs["sa"] = output_file
        return cmd, outputs

    if calc_type == ZeoppCalculationType.volume:
        num_samples = request.num_samples if request.num_samples is not None else 50000
        output_file = output_dir / f"{Path(cif_path).stem}.vol"
        cmd = (
            base_cmd
            + ["-vol", str(chan_radius), str(request.probe_radius), str(num_samples)]
            + extra_args
            + [str(output_file), str(cif_path)]
        )
        outputs["vol"] = output_file
        return cmd, outputs

    if calc_type == ZeoppCalculationType.channel_analysis:
        if request.probe_radius is None:
            raise ValueError("probe_radius is required for channel analysis")
        output_file = output_dir / f"{Path(cif_path).stem}.chan"
        cmd = (
            base_cmd
            + ["-chan", str(request.probe_radius)]
            + extra_args
            + [str(output_file), str(cif_path)]
        )
        outputs["chan"] = output_file
        return cmd, outputs

    raise ValueError(f"Unsupported calculation type: {calc_type}")


def run_zeopp(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    """Execute Zeo++ command."""
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def parse_pore_res(path: Path) -> PoreDiameterResult:
    """Parse Zeo++ .res output for pore diameters."""
    if not path.exists():
        raise FileNotFoundError(f"Missing Zeo++ .res output: {path}")

    content = path.read_text().strip().splitlines()
    for line in content:
        parts = line.strip().split()
        if len(parts) >= 4:
            try:
                return PoreDiameterResult(
                    largest_included_sphere=float(parts[1]),
                    largest_free_sphere=float(parts[2]),
                    largest_included_sphere_along_free_sphere_path=float(parts[3]),
                )
            except ValueError:
                continue
    return PoreDiameterResult()


def parse_sa(path: Path) -> SurfaceAreaResult:
    """Parse Zeo++ .sa surface area output."""
    if not path.exists():
        raise FileNotFoundError(f"Missing Zeo++ .sa output: {path}")

    lines = path.read_text().splitlines()
    for line in lines:
        if not line.startswith("@"):
            continue
        # Example line:
        # @ ... ASA_A^2: 640.105 ASA_m^2/cm^3: 1927.67 ASA_m^2/g: 1336.6 NASA_A^2: 0 NASA_m^2/cm^3: 0 NASA_m^2/g: 0
        tokens = line.split()
        result = SurfaceAreaResult()
        # Case 1: old numeric-only format (6 floats after "@")
        if len(tokens) >= 7 and tokens[1].replace(".", "", 1).replace("-", "", 1).isdigit():
            try:
                floats = [float(x) for x in tokens[1:7]]
                result.asa_a2 = floats[0]
                result.asa_m2_per_cm3 = floats[1]
                result.asa_m2_per_g = floats[2]
                result.nasa_a2 = floats[3]
                result.nasa_m2_per_cm3 = floats[4]
                result.nasa_m2_per_g = floats[5]
                return result
            except Exception:
                pass

        for tok_idx, tok in enumerate(tokens):
            if tok.startswith("ASA_A^2"):
                try:
                    result.asa_a2 = float(tokens[tok_idx + 1].replace(":", ""))
                except Exception:
                    pass
            if tok.startswith("ASA_m^2/cm^3"):
                try:
                    result.asa_m2_per_cm3 = float(tokens[tok_idx + 1].replace(":", ""))
                except Exception:
                    pass
            if tok.startswith("ASA_m^2/g"):
                try:
                    result.asa_m2_per_g = float(tokens[tok_idx + 1].replace(":", ""))
                except Exception:
                    pass
            if tok.startswith("NASA_A^2"):
                try:
                    result.nasa_a2 = float(tokens[tok_idx + 1].replace(":", ""))
                except Exception:
                    pass
            if tok.startswith("NASA_m^2/cm^3"):
                try:
                    result.nasa_m2_per_cm3 = float(tokens[tok_idx + 1].replace(":", ""))
                except Exception:
                    pass
            if tok.startswith("NASA_m^2/g"):
                try:
                    result.nasa_m2_per_g = float(tokens[tok_idx + 1].replace(":", ""))
                except Exception:
                    pass
        return result
    return SurfaceAreaResult()


def parse_vol(path: Path) -> VolumeResult:
    """Parse Zeo++ .vol accessible volume output."""
    if not path.exists():
        raise FileNotFoundError(f"Missing Zeo++ .vol output: {path}")

    lines = path.read_text().splitlines()
    for line in lines:
        if not line.startswith("@"):
            continue
        tokens = line.split()
        result = VolumeResult()
        # Case 1: old numeric-only format (6 floats after "@")
        if len(tokens) >= 7 and tokens[1].replace(".", "", 1).replace("-", "", 1).isdigit():
            try:
                floats = [float(x) for x in tokens[1:7]]
                result.av_a3 = floats[0]
                result.av_fraction = floats[1]
                result.av_cm3_per_g = floats[2]
                result.nav_a3 = floats[3]
                result.nav_fraction = floats[4]
                result.nav_cm3_per_g = floats[5]
                return result
            except Exception:
                pass

        for tok_idx, tok in enumerate(tokens):
            if tok.startswith("AV_A^3"):
                try:
                    result.av_a3 = float(tokens[tok_idx + 1].replace(":", ""))
                except Exception:
                    pass
            if tok.startswith("AV_Volume_fraction") or tok.startswith("AV_fraction"):
                try:
                    result.av_fraction = float(tokens[tok_idx + 1].replace(":", ""))
                except Exception:
                    pass
            if tok.startswith("AV_cm^3/g"):
                try:
                    result.av_cm3_per_g = float(tokens[tok_idx + 1].replace(":", ""))
                except Exception:
                    pass
            if tok.startswith("NAV_A^3"):
                try:
                    result.nav_a3 = float(tokens[tok_idx + 1].replace(":", ""))
                except Exception:
                    pass
            if tok.startswith("NAV_Volume_fraction") or tok.startswith("NAV_fraction"):
                try:
                    result.nav_fraction = float(tokens[tok_idx + 1].replace(":", ""))
                except Exception:
                    pass
            if tok.startswith("NAV_cm^3/g"):
                try:
                    result.nav_cm3_per_g = float(tokens[tok_idx + 1].replace(":", ""))
                except Exception:
                    pass
        return result
    return VolumeResult()


def parse_chan(path: Path, dedup_tolerance: float = 0.0001) -> ChannelAnalysisResult:
    """Parse Zeo++ .chan output for channel analysis."""
    if not path.exists():
        raise FileNotFoundError(f"Missing Zeo++ .chan output: {path}")

    lines = path.read_text().splitlines()
    channel_count = 0
    dimensionalities: List[int] = []
    channels: List[ChannelEntry] = []

    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        header_match = re.search(r"(\d+)\s+channels identified", stripped)
        if header_match:
            try:
                channel_count = int(header_match.group(1))
            except ValueError:
                channel_count = 0
            dim_match = re.search(r"dimensionality\s+([0-9\s]+)", stripped)
            if dim_match:
                dimensionalities = [int(x) for x in dim_match.group(1).split() if x.isdigit()]
            continue
        channel_match = re.match(
            r"^Channel\s+(\d+)\s+([-0-9.eE]+)\s+([-0-9.eE]+)\s+([-0-9.eE]+)",
            stripped,
        )
        if channel_match:
            try:
                idx = int(channel_match.group(1))
                di = float(channel_match.group(2))
                df = float(channel_match.group(3))
                dif = float(channel_match.group(4))
            except ValueError:
                continue
            channels.append(ChannelEntry(index=idx, di=di, df=df, dif=dif))

    if channel_count == 0:
        channel_count = len(channels)

    if dimensionalities and channel_count and len(dimensionalities) == channel_count:
        dim_by_index = {idx: dim for idx, dim in enumerate(dimensionalities)}
        for entry in channels:
            if entry.index in dim_by_index:
                entry.dimensionality = dim_by_index[entry.index]

    if channels:
        deduped: List[ChannelEntry] = []
        for entry in channels:
            merged = False
            for existing in deduped:
                if (existing.dimensionality is None) != (entry.dimensionality is None):
                    continue
                if existing.dimensionality is not None and existing.dimensionality != entry.dimensionality:
                    continue
                if existing.di is None or existing.df is None or existing.dif is None:
                    continue
                if entry.di is None or entry.df is None or entry.dif is None:
                    continue
                if (
                    abs(existing.di - entry.di) <= dedup_tolerance
                    and abs(existing.df - entry.df) <= dedup_tolerance
                    and abs(existing.dif - entry.dif) <= dedup_tolerance
                ):
                    merged = True
                    break
            if not merged:
                deduped.append(entry)
        channels = deduped
        channel_count = len(channels)
        dimensionalities = [c.dimensionality for c in channels if c.dimensionality is not None]

    return ChannelAnalysisResult(
        channel_count=channel_count,
        dimensionalities=dimensionalities,
        channels=channels,
    )


def parse_unitcell_volume(path: Path) -> float:
    """Extract Unitcell_volume from a .vol file."""
    if not path.exists():
        raise FileNotFoundError(f"Missing Zeo++ .vol output: {path}")
    for line in path.read_text().splitlines():
        if line.startswith("@") and "Unitcell_volume:" in line:
            tokens = line.split()
            for idx, tok in enumerate(tokens):
                if tok.startswith("Unitcell_volume"):
                    try:
                        return float(tokens[idx + 1])
                    except Exception as exc:
                        raise ValueError(f"Failed to parse Unitcell_volume in {path}") from exc
    raise ValueError(f"Unitcell_volume not found in {path}")


def get_unitcell_volume(framework: str, probe_radius: float = 1.2, num_samples: int = 50000) -> float:
    """
    Return Unitcell_volume for a framework. Reuse cached .vol if present; otherwise run zeo++ volume.
    """
    settings = get_settings()
    data_root = settings.resolve_path(settings.data_root)
    vol_dir = data_root / "zeopp_runs" / framework.upper()
    vol_path = vol_dir / f"{framework.upper()}.vol"
    if vol_path.exists():
        return parse_unitcell_volume(vol_path)

    req = ZeoppRequest(
        framework=framework,
        calculation_type=ZeoppCalculationType.volume,
        probe_radius=probe_radius,
        chan_radius=probe_radius,
        num_samples=num_samples,
    )
    run_zeopp_request(req)
    return parse_unitcell_volume(vol_path)


def run_zeopp_request(request: ZeoppRequest) -> ZeoppResult:
    """High-level helper to build, run, and parse a Zeo++ calculation."""
    cmd, outputs = build_zeopp_command(request)
    completed = run_zeopp(cmd)

    pore_res: Optional[PoreDiameterResult] = None
    sa_res: Optional[SurfaceAreaResult] = None
    vol_res: Optional[VolumeResult] = None
    chan_res: Optional[ChannelAnalysisResult] = None

    if request.calculation_type == ZeoppCalculationType.pore_diameters:
        pore_res = parse_pore_res(outputs["res"])
    elif request.calculation_type == ZeoppCalculationType.surface_area:
        sa_res = parse_sa(outputs["sa"])
        sa_res.probe_radius = request.probe_radius
    elif request.calculation_type == ZeoppCalculationType.volume:
        vol_res = parse_vol(outputs["vol"])
        vol_res.probe_radius = request.probe_radius
    elif request.calculation_type == ZeoppCalculationType.channel_analysis:
        chan_res = parse_chan(outputs["chan"], request.channel_dedup_tolerance)
        chan_res.probe_radius = request.probe_radius

    raw_output = (completed.stdout or "") + (completed.stderr or "")

    return ZeoppResult(
        framework=request.framework,
        calculation_type=request.calculation_type,
        pore_diameters=pore_res,
        surface_area=sa_res,
        volume=vol_res,
        channel_analysis=chan_res,
        raw_output=raw_output.strip() if raw_output else None,
        output_files={k: str(v) for k, v in outputs.items()},
    )


def zeopp_pore_analysis_tool(
    framework: str,
    probe_radius: Optional[float] = None,
    calculation_type: ZeoppCalculationType = ZeoppCalculationType.pore_diameters,
    chan_radius: Optional[float] = None,
    num_samples: Optional[int] = None,
    extra_args: Optional[List[str]] = None,
    channel_dedup_tolerance: float = 0.0001,
) -> ZeoppResult:
    """
    Convenience entry for agent/tool use. Validates required parameters and runs Zeo++.

    For surface_area/volume runs, probe_radius is required; raises ValueError if missing.
    """
    if calculation_type in (
        ZeoppCalculationType.surface_area,
        ZeoppCalculationType.volume,
        ZeoppCalculationType.channel_analysis,
    ) and (probe_radius is None):
        raise ValueError("probe_radius is required for surface area, volume, or channel analysis calculations.")

    request = ZeoppRequest(
        framework=framework,
        calculation_type=calculation_type,
        probe_radius=probe_radius,
        chan_radius=chan_radius,
        num_samples=num_samples,
        extra_args=extra_args,
        channel_dedup_tolerance=channel_dedup_tolerance,
    )
    return run_zeopp_request(request)


class ZeoppBatchRequest(BaseModel):
    cif_paths: List[str]
    calculation_types: List[ZeoppCalculationType] = [
        ZeoppCalculationType.pore_diameters,
        ZeoppCalculationType.surface_area,
        ZeoppCalculationType.volume,
    ]
    probe_radius: float = 1.2
    chan_radius: Optional[float] = None
    num_samples: Optional[int] = None
    extra_args: Optional[List[str]] = None
    channel_probe_radius: float = 1.5
    channel_dedup_tolerance: float = 0.0001


class ZeoppBatchResultEntry(BaseModel):
    cif_path: str
    results: Dict[str, ZeoppResult] = Field(default_factory=dict)
    errors: Dict[str, str] = Field(default_factory=dict)


class ZeoppBatchResult(BaseModel):
    entries: List[ZeoppBatchResultEntry]


def run_zeopp_batch(request: ZeoppBatchRequest) -> ZeoppBatchResult:
    """Run zeo++ for multiple CIFs and calculation types."""
    entries: List[ZeoppBatchResultEntry] = []
    expanded_paths: List[str] = []
    for cif_path in request.cif_paths:
        if any(ch in cif_path for ch in "*?[]"):
            matches = sorted(glob.glob(cif_path))
            if matches:
                expanded_paths.extend(matches)
            else:
                expanded_paths.append(cif_path)
        else:
            expanded_paths.append(cif_path)
    for cif_path in expanded_paths:
        results: Dict[str, ZeoppResult] = {}
        errors: Dict[str, str] = {}
        for calc in request.calculation_types:
            try:
                if calc == ZeoppCalculationType.channel_analysis:
                    probe_radius = request.channel_probe_radius
                else:
                    probe_radius = request.probe_radius if calc != ZeoppCalculationType.pore_diameters else None
                zeo_req = ZeoppRequest(
                    framework=Path(cif_path).stem,
                    cif_path=cif_path,
                    calculation_type=calc,
                    probe_radius=probe_radius,
                    chan_radius=request.chan_radius,
                    num_samples=request.num_samples,
                    extra_args=request.extra_args,
                    channel_dedup_tolerance=request.channel_dedup_tolerance,
                )
                res = run_zeopp_request(zeo_req)
                results[calc.value] = res
            except Exception as exc:
                errors[calc.value] = str(exc)
        entries.append(ZeoppBatchResultEntry(cif_path=cif_path, results=results, errors=errors))
    return ZeoppBatchResult(entries=entries)


__all__ = [
    "ZeoppCalculationType",
    "ZeoppRequest",
    "PoreDiameterResult",
    "SurfaceAreaResult",
    "VolumeResult",
    "ChannelEntry",
    "ChannelAnalysisResult",
    "ZeoppResult",
    "build_zeopp_command",
    "run_zeopp",
    "parse_pore_res",
    "parse_sa",
    "parse_vol",
    "parse_chan",
    "run_zeopp_request",
    "zeopp_pore_analysis_tool",
    "ZeoppBatchRequest",
    "ZeoppBatchResultEntry",
    "ZeoppBatchResult",
    "run_zeopp_batch",
]
