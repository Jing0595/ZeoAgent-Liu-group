
"""LangGraph-driven ZeoAgent with replanning orchestration and generation subgraph."""

from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypedDict

from langchain_core.language_models import BaseLanguageModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from openai import OpenAI
from pydantic import BaseModel, Field, ValidationError, field_validator

from zeoagent.config import get_settings
from zeoagent.tools import diffusion_predictor
from zeoagent.tools import hpc_generator
from zeoagent.tools import gulp_opt
from zeoagent.tools import iza_match
from zeoagent.tools import ring_tools
from zeoagent.tools import separation_support
from zeoagent.tools import zeopp
from zeoagent.tools.cif_resolver import resolve_cif_path


SYSTEM_PROMPT = (
    "You are ZeoAgent, a zeolite assistant with a planner-executor-critic-finalizer loop. "
    "Use tool evidence only and do not invent values."
)


@dataclass
class ToolTrace:
    tool: str
    status: str
    input: Dict[str, Any]
    output: Any
    error: Optional[str] = None
    loop: int = 0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tool": self.tool,
            "status": self.status,
            "input": self.input,
            "output": self.output,
            "error": self.error,
            "loop": self.loop,
        }


@dataclass
class AgentState:
    """Conversation state persisted across turns."""

    messages: List[Dict[str, str]] = field(default_factory=list)
    memory: Dict[str, Any] = field(default_factory=dict)
    traces: List[ToolTrace] = field(default_factory=list)

    def snapshot(self) -> Dict[str, Any]:
        return {
            "messages": self.messages,
            "memory": self.memory,
            "traces": [t.as_dict() for t in self.traces],
        }


class GraphState(TypedDict):
    messages: List[BaseMessage]
    memory: Dict[str, Any]
    traces: List[ToolTrace]
    intent: Optional[str]
    intent_reason: Optional[str]
    plan: Optional[Dict[str, Any]]
    plan_obj: Optional["Plan"]
    plan_raw: Optional[str]
    plan_error: Optional[str]
    missing_fields: List[str]
    missing_inputs: List[str]
    loop_count: int
    max_loops: int
    critic_decision: Optional[str]
    critic_reason: Optional[str]
    answer: Optional[str]
    final_answer: Optional[str]
    last_tool: Optional[str]
    tool_payload: Dict[str, Any]
    search_results: List[Any]
    reasoning: List[str]


class PlanStep(BaseModel):
    tool: str
    args: Dict[str, Any] = Field(default_factory=dict)
    expected_outputs: List[str] = Field(default_factory=list)
    success_criteria: str

    @field_validator("expected_outputs", mode="before")
    @classmethod
    def _coerce_expected_outputs(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, str):
            return [value]
        return list(value)

    @field_validator("expected_outputs")
    @classmethod
    def _require_expected_outputs(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("expected_outputs is required")
        return value

    @field_validator("success_criteria")
    @classmethod
    def _require_success_criteria(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("success_criteria is required")
        return value


class Plan(BaseModel):
    goal: str
    steps: List[PlanStep]

    @field_validator("steps")
    @classmethod
    def _require_steps(cls, value: List[PlanStep]) -> List[PlanStep]:
        if not value:
            raise ValueError("Plan must contain at least one step")
        return value


class GenerationPlan(BaseModel):
    steps: List[PlanStep]

    @field_validator("steps")
    @classmethod
    def _require_steps(cls, value: List[PlanStep]) -> List[PlanStep]:
        if not value:
            raise ValueError("Generation plan must contain at least one step")
        return value


class DiffusionRequest(BaseModel):
    framework: Optional[str] = None
    temperature_K: Optional[float] = None
    loading_per_uc: Optional[float] = None
    unitcell_volume_a3: Optional[float] = None


@dataclass
class ToolSpec:
    name: str
    input_model: type[BaseModel]
    handler: Callable[[BaseModel], Any]
    description: str


class GenerationSubgraphRequest(BaseModel):
    user_request: str = ""
    dissatisfied: bool = False
    messages: List[Dict[str, str]] = Field(default_factory=list)
    planner_guidance: Optional[str] = None
    excluded_frameworks: List[str] = Field(default_factory=list)


class GenerationCandidate(BaseModel):
    cif_path: str
    ring_types: List[int] = Field(default_factory=list)
    max_ring: int = 0
    ring_metrics: List[Dict[str, float]] = Field(default_factory=list)
    zeopp: Dict[str, Any] = Field(default_factory=dict)
    iza_code: Optional[str] = None
    iza_status: Optional[str] = None


class GenerationResult(BaseModel):
    status: str
    candidates: List[GenerationCandidate] = Field(default_factory=list)
    output_dir: Optional[str] = None
    loop_count: int = 0
    failure_reason: Optional[str] = None
    traces: List[Dict[str, Any]] = Field(default_factory=list)
    reference_lookup: Optional[Dict[str, Any]] = None


class FrameworkReferenceLookupRequest(BaseModel):
    user_request: str = ""


class FrameworkReferenceResult(BaseModel):
    framework: str
    reason: str
    source: str = "llm"
    permeate_species: Optional[str] = None
    validation_hints: List[Dict[str, str]] = Field(default_factory=list)

    @field_validator("framework")
    @classmethod
    def _normalize_framework(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("framework is required")
        return cleaned.upper()

    @field_validator("reason")
    @classmethod
    def _require_reason(cls, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("reason is required")
        return value.strip()

    @field_validator("permeate_species")
    @classmethod
    def _normalize_permeate_species(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        species = separation_support.canonicalize_species(value)
        return species or None

    @field_validator("validation_hints", mode="before")
    @classmethod
    def _normalize_validation_hints(cls, value: Any) -> List[Dict[str, str]]:
        if value is None:
            return []
        if not isinstance(value, list):
            return []
        normalized: List[Dict[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool") or "").strip()
            claim = str(item.get("claim") or "").strip()
            observable = str(item.get("observable") or "").strip()
            criterion = str(item.get("criterion") or "").strip()
            if not (tool and claim and observable and criterion):
                continue
            normalized.append(
                {
                    "tool": tool,
                    "claim": claim,
                    "observable": observable,
                    "criterion": criterion,
                }
            )
        return normalized


TraceWriter = Optional[Callable[[Dict[str, Any]], None]]


def _append_trace(traces: List[ToolTrace], trace: ToolTrace, writer: TraceWriter) -> None:
    if not _keep_trace_raw():
        trace.output = _strip_raw_fields(trace.output)
    traces.append(trace)
    if writer:
        writer(trace.as_dict())


def _emit_trace_event(writer: TraceWriter, event: Dict[str, Any]) -> None:
    if writer:
        writer(event)


def _safe_model_dump(obj: Any, exclude: Optional[set[str]] = None) -> Any:
    if hasattr(obj, "model_dump"):
        try:
            return obj.model_dump(exclude=exclude)
        except TypeError:
            return obj.model_dump()
    if isinstance(obj, list):
        return [_safe_model_dump(item, exclude=exclude) for item in obj]
    if isinstance(obj, dict):
        return {key: _safe_model_dump(value, exclude=exclude) for key, value in obj.items()}
    return obj


def _keep_trace_raw() -> bool:
    return bool(get_settings().debug_trace_raw)


def _strip_raw_fields(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _strip_raw_fields(val) for key, val in value.items() if key != "raw"}
    if isinstance(value, list):
        return [_strip_raw_fields(item) for item in value]
    return value


def _summarize_generation_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    return {
        "status": payload.get("status"),
        "candidates": payload.get("candidates", []),
        "output_dir": payload.get("output_dir"),
        "loop_count": payload.get("loop_count", 0),
    }


def _latest_user_text(messages: List[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content or ""
    return messages[-1].content if messages else ""


def _serialize_messages(messages: List[Any]) -> List[Dict[str, Any]]:
    role_map = {"human": "user", "ai": "assistant", "system": "system", "tool": "tool"}
    serialized: List[Dict[str, Any]] = []
    for msg in messages:
        if isinstance(msg, BaseMessage):
            role = role_map.get(msg.type, getattr(msg, "role", msg.__class__.__name__.lower()))
            serialized.append({"role": role, "content": msg.content})
        elif isinstance(msg, dict):
            serialized.append(msg)
        else:
            serialized.append({"role": "unknown", "content": str(msg)})
    return serialized


_FRAMEWORK_STOPWORDS = {
    "WHAT",
    "IS",
    "THE",
    "OF",
    "AT",
    "WITH",
    "PER",
    "UNIT",
    "CELL",
    "DIFFUSION",
    "COEFFICIENT",
    "SURFACE",
    "AREA",
    "ACCESSIBLE",
    "VOLUME",
    "ETHENE",
    "MOLECULE",
    "MOLECULES",
    "PORE",
    "PORES",
    "DIAMETER",
    "DIAMETERS",
    "K",
}

_KNOWN_FRAMEWORKS_CACHE: Optional[set[str]] = None


def _known_frameworks() -> Optional[set[str]]:
    global _KNOWN_FRAMEWORKS_CACHE
    if _KNOWN_FRAMEWORKS_CACHE is not None:
        return _KNOWN_FRAMEWORKS_CACHE
    try:
        settings = get_settings()
        cif_root = settings.resolve_path(settings.cif_dir)
        if not cif_root.exists() or not cif_root.is_dir():
            return None
        _KNOWN_FRAMEWORKS_CACHE = {p.stem.upper() for p in cif_root.glob("*.cif")}
        return _KNOWN_FRAMEWORKS_CACHE
    except Exception:
        return None


def _normalize_frameworks(values: Iterable[str]) -> List[str]:
    cleaned: List[str] = []
    for raw in values:
        token = str(raw).strip().upper()
        if not token or token in _FRAMEWORK_STOPWORDS:
            continue
        if not token.isalpha() or len(token) < 2 or len(token) > 5:
            continue
        if token not in cleaned:
            cleaned.append(token)
    return cleaned


def _infer_framework(text: str) -> Optional[str]:
    """
    Heuristic to infer framework code from text:
    - prefer 2-5 letter uppercase tokens (e.g., CHA, RHO, MFI)
    - ignore common stopwords
    """
    candidates = re.findall(r"\b([A-Z]{2,5})\b", text or "")
    filtered = _normalize_frameworks(candidates)
    known = _known_frameworks()
    if known:
        for token in filtered:
            if token in known:
                return token
        return None
    if filtered:
        return filtered[0]

    for token in (text or "").split():
        cleaned = token.strip("?.!,").upper()
        if cleaned.isalpha() and 2 <= len(cleaned) <= 5 and cleaned not in _FRAMEWORK_STOPWORDS:
            if known:
                if cleaned in known:
                    return cleaned
            else:
                return cleaned
    return None


def _infer_frameworks(text: str) -> List[str]:
    """
    Heuristic to infer multiple framework codes from text.
    """
    frameworks: List[str] = []
    candidates = re.findall(r"\b([A-Z]{2,5})\b", text or "")
    frameworks.extend(_normalize_frameworks(candidates))
    if frameworks:
        return frameworks
    single = _infer_framework(text)
    return [single] if single else []

def _resolve_framework(text: str, memory: Dict[str, Any]) -> Optional[str]:
    framework = _infer_framework(text)
    if framework:
        memory["framework"] = framework
    return framework


def _update_memory_frameworks(text: str, memory: Dict[str, Any]) -> None:
    frameworks = _infer_frameworks(text)
    existing = memory.get("frameworks")
    merged: List[str] = []
    if isinstance(existing, list):
        merged.extend(_normalize_frameworks(existing))
    if frameworks:
        merged.extend(_normalize_frameworks(frameworks))
    merged = _normalize_frameworks(merged)
    if merged:
        memory["frameworks"] = merged


def _autofill_ring_size_frameworks(
    data: Dict[str, Any], user_request: str, memory: Dict[str, Any]
) -> Dict[str, Any]:
    inferred: List[str] = []
    inferred.extend(_infer_frameworks(user_request))
    mem_frameworks = memory.get("frameworks")
    if isinstance(mem_frameworks, list):
        inferred.extend(mem_frameworks)
    mem_framework = memory.get("framework")
    if mem_framework:
        inferred.append(mem_framework)
    inferred = _normalize_frameworks(inferred)
    steps = data.get("steps")
    if not isinstance(steps, list):
        return data
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("tool") != "ring_size_calculator":
            continue
        args = step.get("args") if isinstance(step.get("args"), dict) else {}
        frameworks = args.get("frameworks")
        if frameworks:
            continue
        framework_arg = args.get("framework")
        pool = list(inferred)
        if framework_arg:
            pool.extend([framework_arg])
        pool = _normalize_frameworks(pool)
        if pool:
            args["frameworks"] = pool
            step["args"] = args
    return data


def _inject_ring_size_frameworks(plan: Plan, user_request: str, memory: Dict[str, Any]) -> None:
    pool: List[str] = []
    pool.extend(_infer_frameworks(user_request))
    mem_frameworks = memory.get("frameworks")
    if isinstance(mem_frameworks, list):
        pool.extend(mem_frameworks)
    mem_framework = memory.get("framework")
    if mem_framework:
        pool.append(mem_framework)
    pool = _normalize_frameworks(pool)
    if not pool:
        return
    for step in plan.steps:
        if step.tool != "ring_size_calculator":
            continue
        frameworks = step.args.get("frameworks")
        if isinstance(frameworks, list) and frameworks:
            continue
        framework_arg = step.args.get("framework")
        final_pool = list(pool)
        if framework_arg:
            final_pool.extend([framework_arg])
        final_pool = _normalize_frameworks(final_pool)
        if final_pool:
            step.args["frameworks"] = final_pool


def _tool_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def _requires_novelty(user_request: str) -> bool:
    if not user_request:
        return False
    text = user_request.lower()
    keywords = (
        "hypothetical",
        "novel",
        "not in iza",
        "not in the iza",
        "non-iza",
        "not in database",
        "not in the database",
        "unknown to iza",
        "unreported in iza",
    )
    return any(key in text for key in keywords)


def _requires_channel_analysis(user_request: str) -> bool:
    if not user_request:
        return False
    text = user_request.lower()
    if re.search(r"\b[123]\s*[-–]?\s*d\b", text):
        return True
    if "one dimensional" in text or "two dimensional" in text or "three dimensional" in text:
        return True
    keywords = (
        "channel",
        "channels",
        "dimensionality",
        "accessible",
        "accessibility",
    )
    return any(key in text for key in keywords)


def _preferred_ring_validation(user_request: str) -> Optional[str]:
    if not user_request:
        return None
    text = user_request.lower()
    if "crum" in text and any(token in text for token in ("use", "using", "validation", "method", "validate")):
        return "crum"
    if "sastre" in text and any(token in text for token in ("use", "using", "validation", "method", "validate")):
        return "sastre"
    return None


def _sanitize_reference_lookup_user_request(user_request: str) -> str:
    """
    Remove ring-validation directives from reference lookup prompts so tokens like
    'crum' are not misread as framework hints (e.g., CRU).
    """
    text = (user_request or "").strip()
    if not text:
        return ""
    mode = r"(?:auto|sastre|crum|none|null|false|off)"
    patterns = [
        rf"""(?ix)
        (?:^|[\s,，;；\(\)\[\]\{{\}}])
        validation(?:\s+method)?\s*[:=]\s*["']?{mode}["']?
        """,
        rf"""(?ix)
        (?:^|[\s,，;；\(\)\[\]\{{\}}])
        (?:use|using|with)\s+["']?{mode}["']?\s+(?:validation(?:\s+method)?|method)
        """,
        rf"""(?ix)
        (?:^|[\s,，;；\(\)\[\]\{{\}}])
        validation(?:\s+method)?\s+(?:is\s+)?["']?{mode}["']?
        """,
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"\s*([,，;；:：])\s*", r"\1 ", cleaned)
    cleaned = cleaned.strip(" ,，;；:：")
    return cleaned or text


def _public_separation_context(separation_context: Dict[str, Any]) -> Dict[str, Any]:
    if not separation_context:
        return {"is_separation_task": False}
    return {
        "is_separation_task": bool(separation_context.get("is_separation_task")),
        "gas_pair": separation_context.get("gas_pair"),
        "permeate_species": separation_context.get("permeate_species"),
        "permeate_diameter_A": separation_context.get("permeate_diameter_A"),
        "retentate_species": separation_context.get("retentate_species"),
        "retentate_diameter_A": separation_context.get("retentate_diameter_A"),
        "diameter_source": separation_context.get("diameter_source"),
        "validation_hints": separation_context.get("validation_hints", []),
        "retrieved_chunks": separation_context.get("retrieved_chunks", []),
    }


def _critic_separation_context(separation_context: Dict[str, Any]) -> Dict[str, Any]:
    if not separation_context:
        return {"is_separation_task": False}
    return {
        "is_separation_task": bool(separation_context.get("is_separation_task")),
        "gas_pair": separation_context.get("gas_pair"),
        "permeate_species": separation_context.get("permeate_species"),
        "permeate_diameter_A": separation_context.get("permeate_diameter_A"),
        "retentate_species": separation_context.get("retentate_species"),
        "retentate_diameter_A": separation_context.get("retentate_diameter_A"),
        "diameter_source": separation_context.get("diameter_source"),
        "validation_hints": separation_context.get("validation_hints", []),
    }


def _refresh_separation_retrieval_context(
    user_request: str,
    separation_context: Dict[str, Any],
    force_new: bool = False,
) -> None:
    if not separation_context.get("is_separation_task"):
        return
    corpus_dir_value = separation_context.get("_corpus_dir")
    if not corpus_dir_value:
        return
    used_ids = [
        str(item).strip()
        for item in separation_context.get("_used_retrieval_hit_ids", [])
        if str(item).strip()
    ]
    excluded = used_ids if force_new else []
    hits = separation_support.retrieve_separation_evidence(
        user_request,
        Path(str(corpus_dir_value)),
        top_k=1,
        excluded_hit_ids=excluded,
    )
    if not hits and excluded:
        hits = separation_support.retrieve_separation_evidence(
            user_request,
            Path(str(corpus_dir_value)),
            top_k=1,
        )
    separation_context["retrieved_chunks"] = separation_support.compact_hits_for_trace(hits)
    separation_context["retrieval_prompt"] = separation_support.format_retrieval_context(hits)
    if hits:
        hit_id = separation_support.hit_identifier(hits[0])
        if hit_id and hit_id not in used_ids:
            used_ids.append(hit_id)
    separation_context["_used_retrieval_hit_ids"] = used_ids


def _default_separation_validation_hints(separation_context: Dict[str, Any]) -> List[Dict[str, str]]:
    species = str(separation_context.get("permeate_species") or "permeate").strip().upper()
    diameter = separation_context.get("permeate_diameter_A")
    if diameter is None:
        passability = f"largest_free_sphere should be reported for {species} passability screening."
    else:
        passability = f"largest_free_sphere >= {float(diameter):.2f} A for {species} passability."
    return [
        {
            "tool": "zeopp_batch",
            "claim": f"{species} should be transport-accessible in candidate pores.",
            "observable": "largest_free_sphere",
            "criterion": passability,
        },
        {
            "tool": "ring_type_calculator",
            "claim": "Candidate topology should remain consistent with reference-driven sieving rationale.",
            "observable": "ring_types,max_ring",
            "criterion": "Use ring_types and max_ring to verify similarity with reference topology constraints.",
        },
    ]


def _normalize_separation_validation_hints(
    raw_hints: Any,
    separation_context: Dict[str, Any],
) -> List[Dict[str, str]]:
    if not isinstance(raw_hints, list):
        return _default_separation_validation_hints(separation_context)
    hints: List[Dict[str, str]] = []
    for item in raw_hints:
        if not isinstance(item, dict):
            continue
        tool = str(item.get("tool") or "").strip()
        claim = str(item.get("claim") or "").strip()
        observable = str(item.get("observable") or "").strip()
        criterion = str(item.get("criterion") or "").strip()
        if not (tool and claim and observable and criterion):
            continue
        hints.append(
            {
                "tool": tool,
                "claim": claim,
                "observable": observable,
                "criterion": criterion,
            }
        )
    return hints or _default_separation_validation_hints(separation_context)


def _required_separation_tools(
    reference: FrameworkReferenceResult,
    separation_context: Dict[str, Any],
) -> List[str]:
    required: set[str] = set()
    hints = _normalize_separation_validation_hints(reference.validation_hints, separation_context)
    reason_text = (reference.reason or "").lower()
    for hint in hints:
        tool = (hint.get("tool") or "").strip()
        if tool:
            required.add(tool)
        merged = " ".join(
            [hint.get("claim", ""), hint.get("observable", ""), hint.get("criterion", "")]
        ).lower()
        if any(token in merged for token in ("largest_free_sphere", "permeate", "passability", "transport")):
            required.add("zeopp_batch")
        if any(token in merged for token in ("ring", "max_ring", "topolog", "small-pore", "small pore")):
            required.add("ring_type_calculator")
    if any(token in reason_text for token in ("ring", "max_ring", "small-pore", "small pore", "window")):
        required.add("ring_type_calculator")
    return sorted(required)


def _build_separation_planner_overlay(
    reference: FrameworkReferenceResult,
    separation_context: Dict[str, Any],
) -> str:
    hints = _normalize_separation_validation_hints(reference.validation_hints, separation_context)
    hints_json = json.dumps(hints, ensure_ascii=False)
    gas_pair = separation_context.get("gas_pair") or "unknown"
    species = separation_context.get("permeate_species") or "unknown"
    diameter = separation_context.get("permeate_diameter_A")
    diameter_text = "unknown" if diameter is None else f"{float(diameter):.2f} A"
    return (
        "Separation-specific planning overlay:\n"
        f"- gas_pair: {gas_pair}\n"
        f"- permeate_species: {species}\n"
        f"- permeate_diameter_A: {diameter_text}\n"
        f"- reference_reason: {reference.reason}\n"
        f"- validation_hints: {hints_json}\n"
        "Plan by translating each validation hint into executable tool steps and measurable success_criteria. "
        "Do not rely on fixed templates; derive checks from the selected reference rationale and hints. "
        "For each hint, include at least one step whose outputs can directly verify the criterion."
    )


def _validate_separation_plan_coverage(
    plan: GenerationPlan,
    reference: FrameworkReferenceResult,
    separation_context: Optional[Dict[str, Any]],
) -> Optional[str]:
    if not separation_context or not separation_context.get("is_separation_task"):
        return None
    required_tools = _required_separation_tools(reference, separation_context)
    plan_tools = [step.tool for step in plan.steps]
    missing = [tool for tool in required_tools if tool not in plan_tools]
    if missing:
        return (
            "Separation plan missing validation tools implied by reference rationale: "
            + ", ".join(missing)
        )
    if "zeopp_batch" in required_tools:
        has_pore_diameter = False
        for step in plan.steps:
            if step.tool != "zeopp_batch":
                continue
            calc_types = step.args.get("calculation_types")
            if isinstance(calc_types, list) and "pore_diameters" in calc_types:
                has_pore_diameter = True
                break
        if not has_pore_diameter:
            return "Separation plan requires zeopp_batch with pore_diameters for passability validation."
    return None


def _build_separation_context(user_request: str) -> Dict[str, Any]:
    if not separation_support.is_separation_request(user_request):
        return {"is_separation_task": False}

    settings = get_settings()
    data_root = settings.resolve_path(settings.data_root)
    diameter_path = data_root / "knowledge" / "molecular_kinetic_diameters.json"
    corpus_dir = data_root / "corpus"

    diameter_table = separation_support.load_molecular_diameter_table(diameter_path)
    gas_pair = separation_support.infer_gas_pair(user_request)
    permeate_species = separation_support.infer_permeate_species(user_request, diameter_table)
    retentate_species = separation_support.infer_retentate_species(
        user_request,
        diameter_table,
        permeate_species=permeate_species,
    )
    diameter_entry = separation_support.resolve_species_diameter(permeate_species, diameter_table)
    retentate_entry = separation_support.resolve_species_diameter(retentate_species, diameter_table)
    context = {
        "is_separation_task": True,
        "gas_pair": gas_pair,
        "permeate_species": permeate_species,
        "permeate_diameter_A": (
            float(diameter_entry["kinetic_diameter_A"]) if diameter_entry else None
        ),
        "retentate_species": retentate_species,
        "retentate_diameter_A": (
            float(retentate_entry["kinetic_diameter_A"]) if retentate_entry else None
        ),
        "diameter_source": diameter_entry.get("source") if diameter_entry else None,
        "retrieved_chunks": [],
        "retrieval_prompt": "",
        "_corpus_dir": str(corpus_dir),
        "_used_retrieval_hit_ids": [],
        "_user_request": user_request,
        "_diameter_table": diameter_table,
    }
    context["validation_hints"] = _default_separation_validation_hints(context)
    _refresh_separation_retrieval_context(user_request, context, force_new=False)
    return context


def _update_separation_context_from_lookup(
    separation_context: Dict[str, Any],
    lookup_result: FrameworkReferenceResult,
) -> None:
    if not separation_context.get("is_separation_task"):
        return
    species = None
    if lookup_result.permeate_species:
        species = separation_support.canonicalize_species(lookup_result.permeate_species)
    if species:
        separation_context["permeate_species"] = species
        table = separation_context.get("_diameter_table", {})
        entry = separation_support.resolve_species_diameter(species, table)
        if entry:
            separation_context["permeate_diameter_A"] = float(entry["kinetic_diameter_A"])
            separation_context["diameter_source"] = entry.get("source")
        user_request = str(separation_context.get("_user_request") or "")
        retentate = separation_support.infer_retentate_species(
            user_request,
            table if isinstance(table, dict) else {},
            permeate_species=species,
        )
        ret_entry = separation_support.resolve_species_diameter(
            retentate,
            table if isinstance(table, dict) else {},
        )
        separation_context["retentate_species"] = retentate
        separation_context["retentate_diameter_A"] = (
            float(ret_entry["kinetic_diameter_A"]) if ret_entry else None
        )
    separation_context["validation_hints"] = _normalize_separation_validation_hints(
        lookup_result.validation_hints,
        separation_context,
    )


def _candidate_largest_free_sphere(candidate: GenerationCandidate) -> Optional[float]:
    try:
        value = (
            candidate.zeopp.get("results", {})
            .get("pore_diameters", {})
            .get("pore_diameters", {})
            .get("largest_free_sphere")
        )
    except Exception:
        value = None
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _single_cif_dir(cif_path: Path, label: str) -> Path:
    settings = get_settings()
    base_dir = settings.resolve_path(settings.data_root) / "tool_cif_dirs"
    run_dir = base_dir / f"{label}-{_tool_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cif_path, run_dir / cif_path.name)
    return run_dir


def _multi_cif_dir(frameworks: List[str], label: str) -> Path:
    settings = get_settings()
    base_dir = settings.resolve_path(settings.data_root) / "tool_cif_dirs"
    run_dir = base_dir / f"{label}-{_tool_timestamp()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    for framework in frameworks:
        cif_path = resolve_cif_path(framework)
        shutil.copy2(cif_path, run_dir / cif_path.name)
    return run_dir


def _resolve_cif_dir_for_tool(
    cif_dir: Optional[str],
    framework: Optional[str],
    fallback_framework: Optional[str],
    label: str,
) -> Optional[str]:
    if cif_dir:
        return cif_dir
    resolved_framework = framework or fallback_framework
    if not resolved_framework:
        return None
    cif_path = resolve_cif_path(resolved_framework)
    return str(_single_cif_dir(Path(cif_path), label))


def _extract_temp_loading(text: str, memory: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    temperature = memory.get("temperature_K")
    loading = memory.get("loading_per_uc")

    if temperature is None:
        temp_match = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:K|kelvin)\b", text, re.IGNORECASE)
        if temp_match:
            temperature = float(temp_match.group(1))
            memory["temperature_K"] = temperature

    if loading is None:
        loading_match = re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s+(?:ethene|molecule|molecules|adsorbate)s?\s+per\s+unit\s+cell",
            text,
            re.IGNORECASE,
        )
        if loading_match:
            loading = float(loading_match.group(1))
            memory["loading_per_uc"] = loading

    return temperature, loading


def _is_generation_intent(text: str) -> bool:
    lower = text.lower()
    keywords = [
        "generate",
        "design",
        "new structure",
        "new framework",
        "synthesize",
        "create",
        "modify",
        "adapt",
        "derive",
        "derivative",
        "variant",
        "alter",
        "i want a hypothetical",
    ]
    return any(k in lower for k in keywords)


def _is_parameter_update(text: str) -> bool:
    has_temp = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*(?:K|kelvin)\b", text, re.IGNORECASE) is not None
    has_loading = (
        re.search(
            r"([0-9]+(?:\.[0-9]+)?)\s+(?:ethene|molecule|molecules|adsorbate)s?\s+per\s+unit\s+cell",
            text,
            re.IGNORECASE,
        )
        is not None
    )
    if not (has_temp or has_loading):
        return False
    return not _is_generation_intent(text)


def _is_dissatisfied(text: str) -> bool:
    lower = text.lower()
    return any(
        phrase in lower
        for phrase in ["not satisfied", "unsatisfied", "not good", "bad", "try again", "another", "dislike"]
    )


def _extract_json_block(raw: str) -> str:
    text = raw.strip()
    if "```" in text:
        fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fence_match:
            return fence_match.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1]
    return text


def _extract_json_objects(raw: str) -> List[Dict[str, Any]]:
    text = raw.strip()
    if "```" in text:
        fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1)
    decoder = json.JSONDecoder()
    objects: List[Dict[str, Any]] = []
    idx = 0
    while idx < len(text):
        start = text.find("{", idx)
        if start == -1:
            break
        try:
            obj, end = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            idx = start + 1
            continue
        if isinstance(obj, dict):
            objects.append(obj)
        idx = start + end
    return objects


def _extract_first_json_object(
    raw: str,
    required_keys: Optional[Iterable[str]] = None,
) -> Optional[Dict[str, Any]]:
    candidates = _extract_json_objects(raw)
    if not candidates:
        return None
    if required_keys:
        required = set(required_keys)
        for candidate in candidates:
            if required.issubset(candidate.keys()):
                return candidate
    return candidates[0]


def _collect_missing_fields(exc: ValidationError) -> List[str]:
    fields: List[str] = []
    for err in exc.errors():
        if err.get("type") == "missing":
            loc = err.get("loc", [])
            fields.append(".".join(str(item) for item in loc))
    return fields


def _apply_plan_defaults(data: Dict[str, Any]) -> Dict[str, Any]:
    steps = data.get("steps")
    if not isinstance(steps, list):
        return data
    for step in steps:
        if not isinstance(step, dict):
            continue
        args = step.get("args")
        if isinstance(args, list):
            converted: Dict[str, Any] = {}
            for item in args:
                if isinstance(item, dict) and "key" in item:
                    converted[str(item["key"])] = item.get("value")
            step["args"] = converted
        success = step.get("success_criteria")
        if isinstance(success, str) and success.strip():
            continue
        tool = step.get("tool") or "step"
        expected = step.get("expected_outputs")
        if isinstance(expected, list) and expected:
            expected_str = ", ".join(str(item) for item in expected)
            step["success_criteria"] = f"{tool} produces {expected_str}"
        elif isinstance(expected, str) and expected.strip():
            step["success_criteria"] = f"{tool} produces {expected}"
        else:
            step["success_criteria"] = f"{tool} completed"
        if step.get("tool") == "ring_size_calculator":
            args = step.get("args")
            if isinstance(args, dict) and not args.get("frameworks"):
                framework_arg = args.get("framework")
                if framework_arg:
                    normalized = _normalize_frameworks([framework_arg])
                    if normalized:
                        args["frameworks"] = normalized
                        step["args"] = args
    return data


def _parse_plan(
    raw: str,
    tools: Dict[str, ToolSpec],
    min_steps: int = 1,
    max_steps: Optional[int] = None,
    user_request: str = "",
    memory: Optional[Dict[str, Any]] = None,
):
    missing_fields: List[str] = []
    data = None
    candidates = _extract_json_objects(raw)
    if candidates:
        for candidate in candidates:
            if "goal" in candidate and "steps" in candidate:
                data = candidate
                break
        if data is None:
            data = candidates[0]
    else:
        try:
            data = json.loads(_extract_json_block(raw))
        except json.JSONDecodeError as exc:
            return None, f"Failed to parse plan JSON: {exc}", missing_fields
    if data is None:
        return None, "Failed to parse plan JSON: no object found", missing_fields
    data = _apply_plan_defaults(data)
    data = _autofill_ring_size_frameworks(data, user_request, memory or {})

    try:
        plan = Plan.model_validate(data)
    except ValidationError as exc:
        missing_fields.extend(_collect_missing_fields(exc))
        return None, f"Plan schema invalid: {exc}", missing_fields

    if min_steps and len(plan.steps) < min_steps:
        return None, f"Plan must include at least {min_steps} steps", missing_fields
    if max_steps is not None and len(plan.steps) > max_steps:
        return None, f"Plan must include at most {max_steps} steps", missing_fields

    for step in plan.steps:
        if step.tool not in tools:
            return None, f"Unknown tool '{step.tool}' in plan", missing_fields
        tool_model = tools[step.tool].input_model
        if step.tool == "generation_subgraph":
            dissatisfied = step.args.get("dissatisfied")
            if dissatisfied is not None and not isinstance(dissatisfied, bool):
                step.args.pop("dissatisfied", None)
            step.args.pop("constraints", None)
        if step.tool == "hpc_generation":
            for key in ("allow_reuse", "num_seeds", "threshold", "retry_on_empty", "retry_num_seeds", "retry_threshold_step"):
                step.args.pop(key, None)
        if step.tool == "ring_size_calculator":
            frameworks = step.args.get("frameworks")
            if frameworks is not None and (not isinstance(frameworks, list) or not frameworks):
                step.args.pop("frameworks", None)
            args_for_validation = dict(step.args)
            args_for_validation.pop("frameworks", None)
        else:
            args_for_validation = step.args
        try:
            tool_model(**args_for_validation)
        except ValidationError as exc:
            missing_fields.extend(_collect_missing_fields(exc))
            return None, f"Invalid args for tool '{step.tool}': {exc}", missing_fields
    return plan, None, missing_fields


def _validate_plan_obj(
    plan: Plan,
    tools: Dict[str, ToolSpec],
    min_steps: int = 1,
    max_steps: Optional[int] = None,
    user_request: str = "",
    memory: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[Plan], Optional[str], List[str]]:
    missing_fields: List[str] = []
    if min_steps and len(plan.steps) < min_steps:
        return None, f"Plan must include at least {min_steps} steps", missing_fields
    if max_steps is not None and len(plan.steps) > max_steps:
        return None, f"Plan must include at most {max_steps} steps", missing_fields

    for step in plan.steps:
        if step.tool not in tools:
            return None, f"Unknown tool '{step.tool}' in plan", missing_fields
        tool_model = tools[step.tool].input_model
        if step.tool == "generation_subgraph":
            dissatisfied = step.args.get("dissatisfied")
            if dissatisfied is not None and not isinstance(dissatisfied, bool):
                step.args.pop("dissatisfied", None)
            step.args.pop("constraints", None)
        if step.tool == "hpc_generation":
            for key in (
                "allow_reuse",
                "num_seeds",
                "threshold",
                "retry_on_empty",
                "retry_num_seeds",
                "retry_threshold_step",
            ):
                step.args.pop(key, None)
        if step.tool == "ring_size_calculator":
            frameworks = step.args.get("frameworks")
            if frameworks is None or (isinstance(frameworks, list) and not frameworks):
                pool: List[str] = []
                pool.extend(_infer_frameworks(user_request))
                mem = memory or {}
                mem_frameworks = mem.get("frameworks")
                if isinstance(mem_frameworks, list):
                    pool.extend(mem_frameworks)
                mem_framework = mem.get("framework")
                if mem_framework:
                    pool.append(mem_framework)
                framework_arg = step.args.get("framework")
                if framework_arg:
                    pool.append(framework_arg)
                pool = _normalize_frameworks(pool)
                if pool:
                    step.args["frameworks"] = pool
                    frameworks = pool
            if frameworks is not None and (not isinstance(frameworks, list) or not frameworks):
                step.args.pop("frameworks", None)
            args_for_validation = dict(step.args)
            args_for_validation.pop("frameworks", None)
        else:
            args_for_validation = step.args
        try:
            tool_model(**args_for_validation)
        except ValidationError as exc:
            missing_fields.extend(_collect_missing_fields(exc))
            return None, f"Invalid args for tool '{step.tool}': {exc}", missing_fields
    return plan, None, missing_fields


def _resolve_placeholders(value: Any, context: Dict[str, Any]) -> Any:
    if isinstance(value, str):
        replacements = {
            "{output_dir}": context.get("output_dir"),
            "{{output_dir}}": context.get("output_dir"),
            "$output_dir": context.get("output_dir"),
            "{cif_dir}": context.get("cif_dir"),
            "{{cif_dir}}": context.get("cif_dir"),
            "$cif_dir": context.get("cif_dir"),
            "{reference_framework}": context.get("reference_framework"),
            "{{reference_framework}}": context.get("reference_framework"),
            "$reference_framework": context.get("reference_framework"),
            "{reference_cif_dir}": context.get("reference_cif_dir"),
            "{{reference_cif_dir}}": context.get("reference_cif_dir"),
            "$reference_cif_dir": context.get("reference_cif_dir"),
            "{optimized_reference_cif_dir}": context.get("reference_cif_dir"),
            "{{optimized_reference_cif_dir}}": context.get("reference_cif_dir"),
            "$optimized_reference_cif_dir": context.get("reference_cif_dir"),
            "{reference_cif_path}": context.get("reference_cif_path"),
            "{{reference_cif_path}}": context.get("reference_cif_path"),
            "$reference_cif_path": context.get("reference_cif_path"),
            "{optimized_reference_cif_path}": context.get("reference_cif_path"),
            "{{optimized_reference_cif_path}}": context.get("reference_cif_path"),
            "$optimized_reference_cif_path": context.get("reference_cif_path"),
        }
        for token, replacement in replacements.items():
            if replacement is None:
                continue
            if value == token:
                return replacement
            if token in value:
                value = value.replace(token, str(replacement))
        return value
    if isinstance(value, list):
        return [_resolve_placeholders(item, context) for item in value]
    if isinstance(value, dict):
        return {k: _resolve_placeholders(v, context) for k, v in value.items()}
    return value


def _build_tool_prompt(tools: Dict[str, ToolSpec]) -> str:
    lines = []
    for spec in tools.values():
        lines.append(f"- {spec.name}: {spec.description}")
    return "\n".join(lines)


def _parse_generation_steps(data: Dict[str, Any]) -> Tuple[Optional[GenerationPlan], Optional[str]]:
    steps = data.get("steps")
    if not isinstance(steps, list) or not steps:
        return None, "steps must be a non-empty list"
    parsed_steps: List[PlanStep] = []
    for step in steps:
        if not isinstance(step, dict):
            return None, "each step must be an object"
        try:
            parsed_steps.append(PlanStep.model_validate(step))
        except ValidationError as exc:
            return None, f"Generation step schema invalid: {exc}"
    return GenerationPlan(steps=parsed_steps), None


def _validate_generation_steps(steps: List[PlanStep]) -> Optional[str]:
    tools = [step.tool for step in steps]
    if "framework_reference_lookup" not in tools:
        return "Plan must include framework_reference_lookup before hpc_generation"
    if "hpc_generation" not in tools:
        return "Plan must include hpc_generation"
    if tools.index("framework_reference_lookup") > tools.index("hpc_generation"):
        return "framework_reference_lookup must precede hpc_generation"
    return None


def _run_generation_plan_responses(
    client: OpenAI,
    model: str,
    user_request: str,
    reference: FrameworkReferenceResult,
    tool_specs: Dict[str, ToolSpec],
    messages_json: Optional[str] = None,
    planner_guidance: Optional[str] = None,
    separation_context: Optional[Dict[str, Any]] = None,
) -> Tuple[Optional[GenerationPlan], str, Optional[str]]:
    prompt = (
        f"{_build_generation_plan_prompt(tool_specs)} Reference selection (already chosen): "
        f"framework={reference.framework}; reason={reference.reason}. "
        "Ensure framework_reference_lookup aligns with this reference."
    )
    if separation_context and separation_context.get("is_separation_task"):
        prompt = f"{prompt}\n\n{_build_separation_planner_overlay(reference, separation_context)}"
    user_content = user_request
    if messages_json:
        user_content = f"User request: {user_request}\nMessages: {messages_json}"
    if planner_guidance:
        user_content = (
            f"User request: {user_request}\n"
            f"Planner guidance (non-binding): {planner_guidance}\n"
            f"Messages: {messages_json or ''}"
        ).strip()
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_content},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "generation_plan_preview_narrow",
                "strict": False,
                "schema": _generation_plan_schema(list(tool_specs.keys())),
            }
        },
        temperature=0.1,
    )
    raw = getattr(response, "output_text", "") or ""
    data = _extract_first_json_object(raw, required_keys=("steps",))
    if data is None:
        try:
            data = json.loads(_extract_json_block(raw))
        except json.JSONDecodeError as exc:
            return None, raw, f"Failed to parse plan JSON: {exc}"
    plan, plan_error = _parse_generation_steps(data)
    if plan_error:
        return None, raw, plan_error
    ordering_error = _validate_generation_steps(plan.steps)
    if ordering_error:
        return plan, raw, ordering_error
    separation_error = _validate_separation_plan_coverage(plan, reference, separation_context)
    if separation_error:
        return plan, raw, separation_error
    return plan, raw, None


def _build_general_responses_client() -> OpenAI:
    settings = get_settings()
    if not settings.gpt5_api_key:
        raise ValueError("GPT5_API_KEY not configured in environment or .env")
    kwargs = {"api_key": settings.gpt5_api_key}
    if settings.generation_base_url:
        kwargs["base_url"] = settings.generation_base_url
    return OpenAI(**kwargs)


def _general_plan_schema(tool_names: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "goal": {"type": "string"},
            "steps": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "tool": {"type": "string", "enum": tool_names},
                        "args": {
                            "type": "array",
                            "minItems": 0,
                            "items": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "key": {"type": "string"},
                                    "value": {"type": ["string", "number", "boolean", "null"]},
                                },
                                "required": ["key", "value"],
                            },
                        },
                        "expected_outputs": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                        "success_criteria": {"type": "string"},
                    },
                    "required": ["tool", "args", "expected_outputs", "success_criteria"],
                },
            },
        },
        "required": ["goal", "steps"],
    }


def _run_general_plan_responses(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_request: str,
    intent: str,
    memory_json: str,
    tools_prompt: str,
    tool_specs: Dict[str, ToolSpec],
    messages_json: Optional[str],
) -> Tuple[Optional[Plan], str, Optional[str], List[str]]:
    schema = _general_plan_schema(list(tool_specs.keys()))
    user_content = (
        f"User request: {user_request}\n"
        f"Intent: {intent}\n"
        f"Memory: {memory_json}\n"
        f"Messages: {messages_json}\n"
        "Args format: list of {key, value} objects.\n"
        f"Tools:\n{tools_prompt}\n"
        "Return JSON only."
    )
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "general_plan",
                "strict": True,
                "schema": schema,
            }
        },
        temperature=0.1,
    )
    raw = getattr(response, "output_text", "") or ""
    data = _extract_first_json_object(raw, required_keys=("goal", "steps"))
    if data is None:
        try:
            data = json.loads(_extract_json_block(raw))
        except json.JSONDecodeError as exc:
            return None, raw, f"Failed to parse plan JSON: {exc}", []
    data = _apply_plan_defaults(data)
    try:
        memory = json.loads(memory_json) if memory_json else {}
    except json.JSONDecodeError:
        memory = {}
    data = _autofill_ring_size_frameworks(data, user_request, memory)
    try:
        plan = Plan.model_validate(data)
    except ValidationError as exc:
        missing = _collect_missing_fields(exc)
        return None, raw, f"Plan schema invalid: {exc}", missing
    plan, plan_error, missing_fields = _validate_plan_obj(
        plan, tool_specs, user_request=user_request, memory=memory
    )
    return plan, raw, plan_error, missing_fields



def build_qwen_llm() -> BaseLanguageModel:
    settings = get_settings()
    if not settings.qwen_api_key:
        raise ValueError("QWEN_API_KEY not configured in environment or .env")
    return ChatOpenAI(
        api_key=settings.qwen_api_key,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        model="qwen-plus",
        temperature=0.2,
        timeout=120,
    )


def build_generation_llm() -> BaseLanguageModel:
    settings = get_settings()
    model_name = (settings.generation_model or "gpt-5").strip()
    if model_name.lower() in {"qwen-plus", "qwen"}:
        if not settings.qwen_api_key:
            raise ValueError("QWEN_API_KEY not configured in environment or .env")
        return ChatOpenAI(
            api_key=settings.qwen_api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model="qwen-plus",
            temperature=0.2,
            timeout=120,
        )
    if not settings.gpt5_api_key:
        raise ValueError("GPT5_API_KEY not configured in environment or .env")
    kwargs: Dict[str, Any] = {
        "api_key": settings.gpt5_api_key,
        "model": model_name,
        "temperature": 0.2,
        "timeout": 120,
    }
    if settings.generation_base_url:
        kwargs["base_url"] = settings.generation_base_url
    return ChatOpenAI(**kwargs)



def build_general_llm() -> BaseLanguageModel:
    'Build the default LLM for general tasks (aligned with GPT-5 settings).'
    return build_generation_llm()

def _classify_intent(
    llm: BaseLanguageModel,
    user_text: str,
    memory: Dict[str, Any],
    messages: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[str, str]:
    if _is_generation_intent(user_text):
        return "generate", "Rule-based generation intent triggered by keywords."
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Classify intent as 'generate' or 'general'. Return JSON with keys 'intent' and 'reason'.",
            ),
            (
                "human",
                "User: {message}\nMemory: {memory}\nMessages: {messages}\nReturn JSON only.",
            ),
        ]
    )
    chain = prompt | llm | StrOutputParser()
    raw = chain.invoke({"message": user_text, "memory": json.dumps(memory), "messages": json.dumps(messages or [], ensure_ascii=False)})
    try:
        parsed = json.loads(_extract_json_block(raw))
        intent = parsed.get("intent", "general")
        reason = parsed.get("reason", raw)
        if intent not in {"generate", "general"}:
            intent = "general"
        return intent, reason
    except json.JSONDecodeError:
        return ("generate" if _is_generation_intent(user_text) else "general"), raw


def _summarize_missing_fields(fields: List[str]) -> str:
    cleaned = sorted(set(f for f in fields if f))
    if not cleaned:
        return ""
    return ", ".join(cleaned)


def _has_generation_candidates(traces: List[ToolTrace]) -> bool:
    for trace in reversed(traces):
        if trace.tool != "generation_subgraph" or trace.status != "success":
            continue
        output = trace.output or {}
        if isinstance(output, dict) and output.get("candidates"):
            return True
    return False


def _tool_diffusion(req: BaseModel) -> diffusion_predictor.DiffusionPrediction:
    data = req.model_dump()
    if not data.get("framework"):
        raise ValueError("framework is required for diffusion prediction")
    return diffusion_predictor.predict_ethene_diffusion(**data)


def _tool_zeopp(req: BaseModel) -> zeopp.ZeoppResult:
    request = zeopp.ZeoppRequest(**req.model_dump())
    return zeopp.run_zeopp_request(request)


def _count_t_sites_from_cif(cif_path: Path) -> Optional[int]:
    try:
        from zse import cif_tools
        _, _, tinds = cif_tools.get_tsites(str(cif_path))
        if tinds:
            return len(tinds)
    except Exception:
        pass
    try:
        content = Path(cif_path).read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return None
    count = 0
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 2:
            continue
        if parts[0].startswith("T") and parts[1].isalpha():
            count += 1
    return count or None


def _build_num_unique_list(t_count: Optional[int]) -> Optional[str]:
    if not t_count or t_count < 1:
        return None
    values = [t_count - 1, t_count, t_count + 1]
    filtered = [str(value) for value in values if value > 0]
    if not filtered:
        return None
    return " ".join(filtered)


def _apply_generation_defaults(
    args: Dict[str, Any],
    loop_count: int,
    reference_num_unique_list: Optional[str] = None,
) -> Dict[str, Any]:
    updated = dict(args)
    if not updated.get("num_seeds"):
        updated["num_seeds"] = 15
    if updated.get("threshold") is None:
        updated["threshold"] = 2.2
    if updated.get("allow_reuse") is None:
        updated["allow_reuse"] = True
    if not updated.get("scale_scan"):
        updated["scale_scan"] = "1.0:1.4:0.1"
    if reference_num_unique_list and not updated.get("num_unique_list"):
        updated["num_unique_list"] = reference_num_unique_list
    return updated


REFERENCE_SYSTEM_PROMPT = (
    "You map the user generation intent to an existing zeolite framework reference (IZA code). "
    "If the user explicitly specifies a similar target, use that target as the reference. "
    "Otherwise choose the closest reference and explain why. "
    "Reason must explicitly justify the chosen framework and mention the framework string. "
    "Do NOT mention any other frameworks or alternatives. "
    "If the request includes a largest-ring constraint, only choose frameworks whose largest ring matches. "
    "A framework that merely contains n-rings does NOT satisfy 'largest ring is n-ring' if larger rings exist. "
    "Do NOT invent new codes or ask for more input. Return JSON only."
)

def _build_generation_plan_prompt(tool_specs: Dict[str, ToolSpec]) -> str:
    tools_prompt = _build_tool_prompt(tool_specs)
    return (
        "You are ZeoAgent's generation subgraph planner for zeolite generation tasks. Return JSON with key steps only. "
        "Steps must include framework_reference_lookup and hpc_generation, with lookup first. "
        "Select tools based on user intent: use ring_type_calculator for ring presence/dominance or similarity, "
        "ring_size_calculator for aperture-size windows(Å), and zeopp_batch for pore metrics "
        "(use largest_included_sphere and largest_free_sphere for large-cage checks; require LIS-LFS >= 3 Å; "
        "largest_free_sphere for transport comparisons). "
        "Include channel_analysis only when the user specifies channel size, dimensionality, or accessible channels; "
        "if a channel size requirement is given, set channel_probe_radius "
        "(1.5=8-ring, 2.0=10-ring, 3.2=12-ring). "
        "Include iza_match when novelty is required. "
        "For gas-separation requests, include zeopp_batch with pore_diameters to evaluate largest_free_sphere passability. "
        "When comparing against a reference, always optimize the reference with gulp_opt and compute reference ring/zeopp metrics before comparing; "
        "do NOT optimize generated candidates. "
        "If the user asks to modify/adapt/derive a structure (or implies similarity), include ring_type_calculator on the reference and on candidates to enforce similarity. "
        "The reference ring_type_calculator MUST use the optimized reference CIF (post-gulp_opt). "
        "Do not impose candidate count requirements unless the user explicitly asks for a number. "
        "Tools:\n"
        f"{tools_prompt}\n"
        "Do not ask for more input or output notes; always return a valid plan using best-effort defaults. "
        "Few-shot examples (reference only, adapt to the user request):\n"
    "Example A:\n"
    "{\"steps\":["
    "{\"tool\":\"framework_reference_lookup\",\"args\":{\"user_request\":\"Generate a hypothetical framework similar to CHA\"},"
    "\"expected_outputs\":[\"framework\",\"reason\"],\"success_criteria\":\"Reference chosen\"},"
    "{\"tool\":\"hpc_generation\",\"args\":{\"framework\":\"{reference_framework}\"},"
    "\"expected_outputs\":[\"fetched_paths\"],\"success_criteria\":\"Candidates generated for CHA reference\"},"
    "{\"tool\":\"ring_type_calculator\",\"args\":{\"cif_dir\":\"{reference_cif_dir}\",\"max_ring\":12},"
    "\"expected_outputs\":[\"ring_types\",\"max_ring\"],"
    "\"success_criteria\":\"Reference ring types captured (max_ring fallback)\"},"
    "{\"tool\":\"ring_type_calculator\",\"args\":{\"cif_dir\":\"{output_dir}\",\"max_ring\":12},"
    "\"expected_outputs\":[\"ring_types\",\"max_ring\"],"
    "\"success_criteria\":\"Ring types computed for similarity check\"},"
    "{\"tool\":\"iza_match\",\"args\":{\"cif_dir\":\"{output_dir}\"},"
    "\"expected_outputs\":[\"iza_matches\"],\"success_criteria\":\"Novelty checked against IZA\"}"
    "]}\n"
    "Example B:\n"
    "{\"steps\":["
    "{\"tool\":\"framework_reference_lookup\",\"args\":{\"user_request\":\"Generate a small-pore large-cage zeolite\"},"
    "\"expected_outputs\":[\"framework\",\"reason\"],\"success_criteria\":\"Reference chosen\"},"
    "{\"tool\":\"hpc_generation\",\"args\":{\"framework\":\"{reference_framework}\"},"
    "\"expected_outputs\":[\"fetched_paths\"],\"success_criteria\":\"Candidates generated from reference\"},"
    "{\"tool\":\"ring_type_calculator\",\"args\":{\"cif_dir\":\"{output_dir}\",\"max_ring\":12},"
    "\"expected_outputs\":[\"ring_types\",\"max_ring\"],"
    "\"success_criteria\":\"Small-pore (8-ring) candidates identified\"},"
    "{\"tool\":\"zeopp_batch\",\"args\":{\"cif_paths\":[],\"calculation_types\":[\"pore_diameters\"]},"
    "\"expected_outputs\":[\"zeopp_results\"],"
    "\"success_criteria\":\"Large-cage supported by largest_included_sphere - largest_free_sphere >= 3 Å\"},"
    "{\"tool\":\"iza_match\",\"args\":{\"cif_dir\":\"{output_dir}\"},"
    "\"expected_outputs\":[\"iza_matches\"],\"success_criteria\":\"Novelty checked against IZA\"}"
    "]}\n"
    "Example C:\n"
    "{\"steps\":["
    "{\"tool\":\"framework_reference_lookup\",\"args\":{\"user_request\":\"I want a zeolite featuring 10-membered-ring pore openings as the dominant accessible channels\"},"
    "\"expected_outputs\":[\"framework\",\"reason\"],\"success_criteria\":\"Reference chosen\"},"
    "{\"tool\":\"hpc_generation\",\"args\":{\"framework\":\"{reference_framework}\"},"
    "\"expected_outputs\":[\"fetched_paths\"],\"success_criteria\":\"Candidates generated from reference\"},"
    "{\"tool\":\"zeopp_batch\",\"args\":{\"cif_paths\":[],\"calculation_types\":[\"channel_analysis\"],\"channel_probe_radius\":2.0},"
    "\"expected_outputs\":[\"zeopp_results\"],\"success_criteria\":\"10-ring accessible channels confirmed; dimensionality reported\"},"
    "{\"tool\":\"ring_type_calculator\",\"args\":{\"cif_dir\":\"{output_dir}\",\"max_ring\":12},"
    "\"expected_outputs\":[\"ring_type_counts\",\"ring_types\",\"max_ring\"],\"success_criteria\":\"10-ring count strictly exceeds 8- and 12-ring (no ties)\"},"
    "{\"tool\":\"iza_match\",\"args\":{\"cif_dir\":\"{output_dir}\"},"
    "\"expected_outputs\":[\"iza_matches\"],\"success_criteria\":\"Novelty checked against IZA\"}"
    "]}")


def _generation_plan_schema(tool_names: List[str]) -> Dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "steps": {
                "type": "array",
                "minItems": 2,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "tool": {"type": "string", "enum": tool_names},
                        "args": {"type": "object", "additionalProperties": True},
                        "expected_outputs": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string"},
                        },
                        "success_criteria": {"type": "string"},
                    },
                    "required": ["tool", "args", "expected_outputs", "success_criteria"],
                },
            },
        },
        "required": ["steps"],
    }


def _build_generation_responses_client() -> OpenAI:
    settings = get_settings()
    if not settings.gpt5_api_key:
        raise ValueError("GPT5_API_KEY not configured in environment or .env")
    if not settings.generation_base_url:
        raise ValueError("GENERATION_BASE_URL not configured in environment or .env")
    return OpenAI(api_key=settings.gpt5_api_key, base_url=settings.generation_base_url)


def _reference_lookup_responses(
    client: OpenAI,
    model: str,
    user_request: str,
    excluded_frameworks: Optional[Iterable[str]] = None,
) -> Tuple[FrameworkReferenceResult, str]:
    separation_lookup = "separation task context:" in (user_request or "").lower()
    schema: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "framework": {"type": "string"},
            "reason": {"type": "string"},
            "permeate_species": {"type": ["string", "null"]},
        },
        "required": ["framework", "reason", "permeate_species"],
    }
    if separation_lookup:
        schema["properties"]["validation_hints"] = {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "tool": {"type": "string"},
                    "claim": {"type": "string"},
                    "observable": {"type": "string"},
                    "criterion": {"type": "string"},
                },
                "required": ["tool", "claim", "observable", "criterion"],
            },
        }
        schema["required"] = ["framework", "reason", "permeate_species", "validation_hints"]

    excluded = [item for item in (excluded_frameworks or []) if item]
    system_prompt = REFERENCE_SYSTEM_PROMPT
    if separation_lookup:
        system_prompt = (
            f"{system_prompt} "
            "This is a generation task for gas separation. "
            "Return permeate_species when determinable (canonical token preferred, e.g., CO2, CH4, N2, H2, O2, CO, HE). "
            "Also return validation_hints as executable checks for planning: each item must include "
            "tool, claim, observable, criterion. "
            "Map passability checks to zeopp_batch using largest_free_sphere and include numeric thresholds when available. "
            "When your reason relies on small-pore/ring selectivity or topology similarity, include a ring_type_calculator hint."
        )
    if excluded:
        excluded_list = ", ".join(sorted(set(excluded)))
        system_prompt = (
            f"{system_prompt} "
            f"Do NOT choose any of these frameworks: {excluded_list}. "
            "If any excluded framework would be your first choice, you MUST choose a different framework not in the excluded list. "
            "Do NOT mention the excluded list or excluded frameworks."
        )
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_request},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "framework_reference",
                "strict": True,
                "schema": schema,
            }
        },
        temperature=0.1,
    )
    raw = getattr(response, "output_text", "") or ""
    data = json.loads(_extract_json_block(raw))
    result = FrameworkReferenceResult.model_validate(data)
    return result, raw


def _run_framework_reference_lookup(
    llm: BaseLanguageModel,
    request: FrameworkReferenceLookupRequest,
    fallback_framework: Optional[str],
    excluded_frameworks: Optional[Iterable[str]] = None,
    separation_context: Optional[Dict[str, Any]] = None,
) -> Tuple[FrameworkReferenceResult, str]:
    excluded = {item for item in (excluded_frameworks or []) if item}
    lookup_user_request = _sanitize_reference_lookup_user_request(request.user_request or "")
    lookup_for_llm = lookup_user_request
    if separation_context and separation_context.get("is_separation_task"):
        retrieval_prompt = separation_context.get("retrieval_prompt") or ""
        gas_pair = separation_context.get("gas_pair") or "unknown"
        inferred = separation_context.get("permeate_species") or "unknown"
        lookup_for_llm = (
            f"{lookup_user_request}\n\n"
            "Separation task context:\n"
            f"- gas_pair: {gas_pair}\n"
            f"- inferred_permeate_species: {inferred}\n"
            "Retrieved local evidence:\n"
            f"{retrieval_prompt or 'n/a'}\n"
            "Use the evidence when selecting reference framework, and include permeate_species plus validation_hints in JSON output."
        )
    explicit = _infer_framework(lookup_user_request)
    if explicit and explicit not in excluded:
        return FrameworkReferenceResult(
            framework=explicit,
            reason=f"User explicitly requested framework {explicit}.",
            source="explicit",
            permeate_species=(
                separation_context.get("permeate_species")
                if separation_context and separation_context.get("is_separation_task")
                else None
            ),
            validation_hints=(
                _normalize_separation_validation_hints(
                    separation_context.get("validation_hints"),
                    separation_context,
                )
                if separation_context and separation_context.get("is_separation_task")
                else []
            ),
        ), ""
    try:
        client = _build_generation_responses_client()
        model = (get_settings().generation_model or "gpt-5").strip()
        result, raw = _reference_lookup_responses(client, model, lookup_for_llm, excluded_frameworks)
        if result.framework in excluded:
            retry_prompt = (
                f"{lookup_for_llm}\n"
                "Your previous output violated the excluded framework constraint. "
                "Choose a different framework not in the excluded list."
            )
            result, raw = _reference_lookup_responses(client, model, retry_prompt, excluded_frameworks)
            if result.framework in excluded:
                raise ValueError("framework_reference_lookup returned excluded framework after retry")
        if separation_context and separation_context.get("is_separation_task") and not result.permeate_species:
            result.permeate_species = separation_context.get("permeate_species")
        if separation_context and separation_context.get("is_separation_task"):
            result.validation_hints = _normalize_separation_validation_hints(
                result.validation_hints,
                separation_context,
            )
        result.source = "responses"
        return result, raw
    except (json.JSONDecodeError, ValidationError, ValueError) as exc:
        fallback = ""
        for candidate in _infer_frameworks(lookup_user_request):
            if candidate and candidate not in excluded:
                fallback = candidate
                break
        if not fallback:
            fallback = fallback_framework or _infer_framework(lookup_user_request) or ""
        if not fallback:
            raise ValueError(f"framework_reference_lookup failed: {exc}") from exc
        if fallback in excluded:
            raise ValueError("framework_reference_lookup fallback hit excluded framework")
        reason = f"Fallback to inferred framework because lookup output was invalid: {exc}"
        result = FrameworkReferenceResult(
            framework=fallback,
            reason=reason,
            source="fallback",
            permeate_species=(
                separation_context.get("permeate_species")
                if separation_context and separation_context.get("is_separation_task")
                else None
            ),
            validation_hints=(
                _normalize_separation_validation_hints(
                    separation_context.get("validation_hints"),
                    separation_context,
                )
                if separation_context and separation_context.get("is_separation_task")
                else []
            ),
        )
        raw = ""
        return result, raw


def _generation_critic_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the semantic critic for zeolite generation. Decide if the candidates satisfy the user intent. "
                "Return JSON with keys: decision (accept|replan) and reason. "
                "When accepting, you may also return selected_cifs (list of CIF filenames or paths) and/or top_k "
                "to indicate which candidates to return. "
                "Rules: "
                "1) Use novelty_required to decide novelty constraints. If novelty_required=false, do NOT reject candidates solely because iza_status=matched. "
                "Only treat novelty as required when the user explicitly requests hypothetical, novel, not in IZA, or not in database. "
                "When novelty is required, any candidate with iza_status=matched MUST be rejected (replan unless other novel candidates satisfy constraints). "
                "2) If iza_status is matched and iza_code equals the reference framework, treat it as NOT a hypothetical/novel structure; "
                "3) When reference_ring_info is provided, treat candidates as similar if their ring_types match the reference; "
                "use max_ring only as a weaker fallback signal. "
                "4) When novel candidates exist (iza_status=new) that satisfy similarity, select them and drop reference-matched ones instead of replanning just because a reference match is present. "
                "5) If reference ring/zeopp baseline metrics are provided, compare candidates to those baselines (optimized reference), not to any candidate matched to the reference. "
                "6) If the user did not request a count limit and multiple candidates satisfy the constraints, return all qualifying candidates (selected_cifs list) rather than top_k=1. "
                "7) If candidates are insufficient, respond with decision=replan. "
                "8) For separation tasks, evaluate consistency with the selected reference_reason and the plan's executable checks/success_criteria; "
                "9) For separation tasks, when plan/rationale implies passability checks, require candidate observables (e.g., largest_free_sphere) to support acceptance; "
                "if such evidence is missing or contradictory, respond with decision=replan. "
                "10) If separation_context includes retentate_species/retentate_diameter_A, also assess whether candidate metrics plausibly support sieving against the larger component "
                "under the stated reference rationale; if metrics clearly contradict that rationale, prefer decision=replan. "
                "Examples (adapt as needed): "
                "ACCEPT example: {{\"decision\":\"accept\",\"selected_cifs\":[\"candidate.cif\"],\"reason\":\"Ring sizes match CHA-like (max_ring=8) and iza_status is new.\"}} "
                "REPLAN example when matched reference: {{\"decision\":\"replan\",\"reason\":\"Candidate matches reference framework (iza_code=CHA, iza_status=matched), not a new structure; need a novel CHA-like candidate.\"}}",
            ),
            (
                "human",
                "User request: {user_request}\n"
                "Novelty required: {novelty_required}\n"
                "Reference reason: {reference_reason}\n"
                "Reference ring info: {reference_ring_info}\n"
                "Reference ring metrics: {reference_ring_metrics}\n"
                "Reference zeopp: {reference_zeopp}\n"
                "Separation context: {separation_context}\n"
                "Plan: {plan}\nLoop: {loop_count}/{max_loops}\n"
                "Candidates: {candidates}\nReturn JSON only.",
            ),
        ]
    )


def _summarize_generation_candidates(candidates: Dict[str, GenerationCandidate]) -> List[Dict[str, Any]]:
    summary: List[Dict[str, Any]] = []
    for cand in candidates.values():
        summary.append(
            {
                "cif": Path(cand.cif_path).name,
                "cif_path": cand.cif_path,
                "ring_types": cand.ring_types,
                "max_ring": cand.max_ring,
                "ring_metrics": cand.ring_metrics,
                "zeopp": cand.zeopp,
                "iza_code": cand.iza_code,
                "iza_status": cand.iza_status,
            }
        )
    return summary


def _parse_generation_critic(raw: str) -> Tuple[str, str, List[str], Optional[int]]:
    try:
        parsed = json.loads(_extract_json_block(raw))
    except json.JSONDecodeError:
        return "accept", "Semantic critic output invalid; accepting candidates.", [], None
    decision = parsed.get("decision", "accept")
    reason = parsed.get("reason", "") or "Semantic critic decision applied."
    if decision not in {"accept", "replan"}:
        return "accept", reason, [], None
    selected = parsed.get("selected_cifs") or parsed.get("selected") or []
    if isinstance(selected, str):
        selected = [selected]
    if not isinstance(selected, list):
        selected = []
    selected_cifs = [str(item) for item in selected if item]
    top_k = parsed.get("top_k")
    if isinstance(top_k, str) and top_k.isdigit():
        top_k = int(top_k)
    if not isinstance(top_k, int):
        top_k = None
    return decision, reason, selected_cifs, top_k


def _invoke_generation_critic_llm(
    llm: BaseLanguageModel,
    prompt: ChatPromptTemplate,
    inputs: Dict[str, Any],
):
    class _CriticOutput(BaseModel):
        decision: str
        reason: Optional[str] = None
        selected_cifs: List[str] = Field(default_factory=list)
        top_k: Optional[int] = None

    if hasattr(llm, "with_structured_output"):
        for method in ("json_mode", "function_calling"):
            try:
                structured = prompt | llm.with_structured_output(
                    _CriticOutput, method=method, strict=True
                )
                result = structured.invoke(inputs)
                if isinstance(result, dict):
                    result = _CriticOutput.model_validate(result)
                raw = json.dumps(result.model_dump(), ensure_ascii=False)
                return raw
            except Exception:
                pass
    return (prompt | llm | StrOutputParser()).invoke(inputs)


def _select_candidates(
    candidates: Dict[str, GenerationCandidate],
    selected_cifs: List[str],
    top_k: Optional[int],
) -> List[GenerationCandidate]:
    if not candidates:
        return []
    if selected_cifs:
        by_path = {cand.cif_path: cand for cand in candidates.values()}
        by_name = {Path(cand.cif_path).name: cand for cand in candidates.values()}
        by_stem = {Path(cand.cif_path).stem: cand for cand in candidates.values()}
        resolved: List[GenerationCandidate] = []
        seen: set[str] = set()
        for item in selected_cifs:
            cand = by_path.get(item) or by_name.get(item) or by_stem.get(item)
            if cand is None:
                continue
            key = cand.cif_path
            if key in seen:
                continue
            seen.add(key)
            resolved.append(cand)
        if resolved:
            if top_k is not None:
                return resolved[: max(0, top_k)]
            return resolved
    if top_k is not None:
        return list(candidates.values())[: max(0, top_k)]
    return list(candidates.values())


def preview_generation_plan(
    llm: BaseLanguageModel,
    request: GenerationSubgraphRequest,
    loop_count: int = 0,
) -> Dict[str, Any]:
    lookup_result: Optional[FrameworkReferenceResult] = None
    lookup_raw: Optional[str] = None
    lookup_error: Optional[str] = None
    plan_error: Optional[str] = None
    raw_plan: str = ""
    plan: Optional[GenerationPlan] = None

    tool_specs = _build_generation_tool_specs(llm, request.user_request)
    fallback_framework = _infer_framework(request.user_request)
    separation_context = _build_separation_context(request.user_request)
    try:
        lookup_req = FrameworkReferenceLookupRequest(user_request=request.user_request)
        lookup_result, lookup_raw = _run_framework_reference_lookup(
            llm,
            lookup_req,
            fallback_framework,
            None,
            separation_context=separation_context,
        )
    except Exception as exc:
        lookup_error = str(exc)

    if lookup_result:
        try:
            client = _build_generation_responses_client()
            model = (get_settings().generation_model or "gpt-5").strip()
            plan, raw_plan, plan_error = _run_generation_plan_responses(
                client,
                model,
                request.user_request,
                lookup_result,
                tool_specs,
                json.dumps(request.messages, ensure_ascii=False) if request.messages else None,
                request.planner_guidance,
                separation_context=separation_context,
            )
        except Exception as exc:
            plan_error = str(exc)
    else:
        plan_error = plan_error or "Reference lookup failed"

    return {
        "plan": plan.model_dump() if plan else None,
        "plan_error": plan_error,
        "missing_fields": [],
        "raw_plan": raw_plan,
        "loop_count": loop_count,
        "reference_lookup": lookup_result.model_dump() if lookup_result else None,
        "reference_lookup_raw": lookup_raw,
        "reference_lookup_error": lookup_error,
    }


def _build_generation_tool_specs(
    llm: BaseLanguageModel, user_request: str
) -> Dict[str, ToolSpec]:
    return {
        "framework_reference_lookup": ToolSpec(
            name="framework_reference_lookup",
            input_model=FrameworkReferenceLookupRequest,
            handler=lambda req: _run_framework_reference_lookup(
                llm,
                req,
                _infer_framework(user_request),
                None,
            )[0],
            description=(
                "Select a known IZA reference framework based on the user request. "
                "Returns framework and reason. Args: user_request."
            ),
        ),
        "hpc_generation": ToolSpec(
            name="hpc_generation",
            input_model=hpc_generator.HPCJobRequest,
            handler=lambda req: hpc_generator.run_hpc_generation(req),
            description=(
                "Manual HPC integration step for candidate generation. "
                "Users should run the public point-cloud workflow on their own HPC environment "
                "and return local candidate CIF files for downstream screening. "
                "Set candidate_dir when candidate CIFs are already available. "
                "Args: framework, candidate_dir."
            ),
        ),
        "ring_type_calculator": ToolSpec(
            name="ring_type_calculator",
            input_model=ring_tools.RingTypeRequest,
            handler=lambda req: ring_tools.run_ring_type_calculator(req),
            description=(
                "Compute ring_type_counts, ring_types (unique set), and max_ring for CIFs. "
                "Use ring_types for ring presence checks and ring_type_counts for dominance checks. "
                "For similarity, prefer ring_types equality over max_ring; use max_ring only as a weaker fallback. "
                "For 'largest ring is N-ring', require max_ring == N and do not use ring_size_calculator for presence. "
                "When 'dominant' is requested with a specified ring type, compare only 8/10/12, and the target ring count must be strictly largest (no ties). "
                "Treat n-ring, n-membered ring, and n-MR as equivalent ring-type expressions. "
                "For cif_dir, use {output_dir} exactly and do not append subdirectories. Args: cif_dir, max_ring."
            ),
        ),
        "ring_size_calculator": ToolSpec(
            name="ring_size_calculator",
            input_model=ring_tools.RingSizeRequest,
            handler=lambda req: ring_tools.run_ring_size_calculator(req),
            description=(
                "Compute ring geometry stats (dmin/dmax, Angstroms) for a target ring size (default 8-ring). "
                "Use for ring size window checks (e.g., 3.5-3.8 A), not for ring presence. "
                "For cif_dir, use {output_dir} exactly and do not append subdirectories. Args: cif_dir, target_ring_size."
            ),
        ),
        "zeopp_batch": ToolSpec(
            name="zeopp_batch",
            input_model=zeopp.ZeoppBatchRequest,
            handler=lambda req: zeopp.run_zeopp_batch(req),
            description=(
                "Compute pore diameters, surface area, volume, or channel analysis for CIFs. "
                "For large-cage checks, use pore_diameters and require largest_included_sphere - largest_free_sphere >= 3 A. "
                "For cage size requests, include pore_diameters and largest_included_sphere. "
                "For pore volume requests, include volume. "
                "For surface area requests, include surface_area. "
                "For diffusion/transport comparisons, you may compare largest_free_sphere (Df). "
                "Include channel_analysis when the user specifies channel size, dimensionality, or accessible channels. "
                "Do not set probe_radius. For channel_analysis, set channel_probe_radius when ring type is specified "
                "as a channel size requirement (1.5=8-ring, 2.0=10-ring, 3.2=12-ring); otherwise omit channel_probe_radius to use the default. "
                "Args: cif_paths, calculation_types."
            ),
        ),
        "gulp_opt": ToolSpec(
            name="gulp_opt",
            input_model=gulp_opt.GulpOptimizationRequest,
            handler=lambda req: gulp_opt.run_gulp_optimization(req),
            description=(
                "Manual reference-optimization step. "
                "The public release does not ship with the private GULP setup; "
                "users should optimize reference CIFs with their own workflow and rerun downstream analysis. "
                "Args: framework or cif_path."
            ),
        ),
        "iza_match": ToolSpec(
            name="iza_match",
            input_model=iza_match.IzaMatchRequest,
            handler=lambda req: iza_match.run_iza_match(req),
            description=(
                "Match CIFs against the IZA database to assess novelty. "
                "For cif_dir, use {output_dir} exactly and do not append subdirectories. "
                "Returns per-CIF match status. Args: cif_dir, iza_dir, rebuild_cache."
            ),
        ),
    }


def run_generation_subgraph(
    llm: BaseLanguageModel,
    request: GenerationSubgraphRequest,
    trace_writer: TraceWriter = None,
) -> GenerationResult:
    tool_specs = _build_generation_tool_specs(llm, request.user_request)
    separation_context = _build_separation_context(request.user_request)

    loop_count = 0
    max_loops = 5
    traces: List[ToolTrace] = []
    last_failure = None
    output_dir = None
    used_references: List[str] = []
    default_params = {
        "num_seeds": 15,
        "threshold": 2.2,
        "scale_scan": "1.0:1.4:0.1",
        "allow_reuse": True,
        "num_unique_list": "reference_based",
    }
    reference_lookup_data: Optional[Dict[str, Any]] = None

    while loop_count < max_loops:
        if separation_context.get("is_separation_task") and loop_count > 0:
            _refresh_separation_retrieval_context(
                request.user_request,
                separation_context,
                force_new=True,
            )
        _append_trace(
            traces,
            ToolTrace(
                tool="generation_context",
                status="info",
                input={
                    "user_request": request.user_request,
                    "loop_count": loop_count,
                    "max_loops": max_loops,
                    "dissatisfied": request.dissatisfied,
                    "default_generation_params": default_params,
                    "allow_reuse_this_loop": True,
                    "planner_guidance": request.planner_guidance,
                    "excluded_frameworks": request.excluded_frameworks,
                    "separation_context": _public_separation_context(separation_context),
                },
                output={},
                loop=loop_count,
            ),
            trace_writer,
        )
        pre_lookup_result: Optional[FrameworkReferenceResult] = None
        pre_lookup_raw: Optional[str] = None
        try:
            lookup_req = FrameworkReferenceLookupRequest(user_request=request.user_request)
            pre_lookup_result, pre_lookup_raw = _run_framework_reference_lookup(
                llm,
                lookup_req,
                _infer_framework(request.user_request),
                used_references,
                separation_context=separation_context,
            )
            pre_lookup_output = pre_lookup_result.model_dump()
            if separation_context.get("is_separation_task"):
                pre_lookup_output["separation_context"] = _public_separation_context(
                    separation_context
                )
            if _keep_trace_raw():
                pre_lookup_output["raw"] = pre_lookup_raw
            _append_trace(
                traces,
                ToolTrace(
                    tool="framework_reference_lookup_preplan",
                    status="success",
                    input=lookup_req.model_dump(),
                    output=pre_lookup_output,
                    loop=loop_count,
                ),
                trace_writer,
            )
        except Exception as exc:
            pre_lookup_result = None
            pre_lookup_raw = None
            _append_trace(
                traces,
                ToolTrace(
                    tool="framework_reference_lookup",
                    status="error",
                    input={"user_request": request.user_request},
                    output=None,
                    error=str(exc),
                    loop=loop_count,
                ),
                trace_writer,
            )

        plan = None
        raw_plan = ""
        plan_error = "Reference lookup failed"
        if pre_lookup_result:
            try:
                client = _build_generation_responses_client()
                model = (get_settings().generation_model or "gpt-5").strip()
                plan, raw_plan, plan_error = _run_generation_plan_responses(
                    client,
                    model,
                    request.user_request,
                    pre_lookup_result,
                    tool_specs,
                    json.dumps(request.messages, ensure_ascii=False) if request.messages else None,
                    request.planner_guidance,
                    separation_context=separation_context,
                )
            except Exception as exc:
                plan_error = str(exc)
        if plan_error is None and plan is not None:
            _append_trace(
                traces,
                ToolTrace(
                    tool="generation_plan",
                    status="success",
                    input={},
                    output={"plan": plan.model_dump()},
                    loop=loop_count,
                ),
                trace_writer,
            )
        if plan_error or plan is None:
            _append_trace(
                traces,
                ToolTrace(
                    tool="generation_plan",
                    status="error",
                    input={},
                    output={"error": plan_error},
                    loop=loop_count,
                ),
                trace_writer,
            )
            last_failure = plan_error or "generation plan invalid"
            loop_count += 1
            continue

        context: Dict[str, Any] = {}
        candidates: Dict[str, GenerationCandidate] = {}
        step_failed = False
        reference_lookup_data = None

        for step in plan.steps:
            args = _resolve_placeholders(step.args, context)
            if step.tool == "framework_reference_lookup":
                try:
                    req = FrameworkReferenceLookupRequest(user_request=request.user_request)
                except ValidationError as exc:
                    _append_trace(
                        traces,
                        ToolTrace(
                            tool=step.tool,
                            status="error",
                            input={"user_request": request.user_request},
                            output=None,
                            error=f"Validation failed: {exc}",
                            loop=loop_count,
                        ),
                        trace_writer,
                    )
                    last_failure = str(exc)
                    step_failed = True
                    break
                lookup_result = pre_lookup_result
                lookup_raw = pre_lookup_raw or ""
                if lookup_result is None:
                    try:
                        lookup_result, lookup_raw = _run_framework_reference_lookup(
                            llm,
                            req,
                            _infer_framework(request.user_request),
                            used_references,
                            separation_context=separation_context,
                        )
                    except Exception as exc:
                        _append_trace(
                            traces,
                            ToolTrace(
                                tool=step.tool,
                                status="error",
                                input=req.model_dump(),
                                output=None,
                                error=str(exc),
                                loop=loop_count,
                            ),
                            trace_writer,
                        )
                        last_failure = str(exc)
                        step_failed = True
                        break

                context["reference_framework"] = lookup_result.framework
                context["reference_reason"] = lookup_result.reason
                _update_separation_context_from_lookup(separation_context, lookup_result)
                try:
                    ref_cif_path = resolve_cif_path(lookup_result.framework)
                    context["reference_cif_path"] = str(ref_cif_path)
                    context["reference_cif_dir"] = str(_single_cif_dir(ref_cif_path, "generation-ref"))
                    t_count = _count_t_sites_from_cif(ref_cif_path)
                    context["reference_num_unique_list"] = _build_num_unique_list(t_count)
                except Exception:
                    context.pop("reference_cif_path", None)
                    context.pop("reference_cif_dir", None)
                    context.pop("reference_num_unique_list", None)
                if lookup_result.framework not in used_references:
                    used_references.append(lookup_result.framework)
                reference_lookup_data = lookup_result.model_dump()
                if separation_context.get("is_separation_task"):
                    reference_lookup_data["separation_context"] = _public_separation_context(
                        separation_context
                    )
                if _keep_trace_raw():
                    reference_lookup_data["raw"] = lookup_raw
                _append_trace(
                    traces,
                    ToolTrace(
                        tool=step.tool,
                        status="success",
                        input=req.model_dump(),
                        output=reference_lookup_data,
                        loop=loop_count,
                    ),
                    trace_writer,
                )
                continue

            if step.tool == "hpc_generation":
                framework_val = args.get("framework")
                if framework_val in ("{reference_framework}", "{{reference_framework}}", "$reference_framework") or not framework_val:
                    args["framework"] = (
                        context.get("reference_framework")
                        or _infer_framework(request.user_request)
                        or framework_val
                    )
                args = _apply_generation_defaults(
                    args, loop_count, context.get("reference_num_unique_list")
                )
                req = hpc_generator.HPCJobRequest(**args)
                try:
                    _emit_trace_event(
                        trace_writer,
                        {
                            "tool": step.tool,
                            "status": "running",
                            "input": req.model_dump(),
                            "loop": loop_count,
                        },
                    )
                    result = hpc_generator.run_hpc_generation(req)
                    base_dir = Path(result.local_output_dir)
                    dedup_dir = base_dir / "deduplicated"
                    output_dir = str(dedup_dir if dedup_dir.exists() else base_dir)
                    context["output_dir"] = output_dir
                    context["cif_dir"] = output_dir
                    context["cif_paths"] = result.fetched_paths
                    candidates = {path: GenerationCandidate(cif_path=path) for path in result.fetched_paths}
                    _append_trace(
                        traces,
                        ToolTrace(
                            tool=step.tool,
                            status="success",
                            input=req.model_dump(),
                            output=result.model_dump(),
                            loop=loop_count,
                        ),
                        trace_writer,
                    )
                except Exception as exc:
                    _append_trace(
                        traces,
                        ToolTrace(
                            tool=step.tool,
                            status="error",
                            input=req.model_dump(),
                            output=None,
                            error=str(exc),
                            loop=loop_count,
                        ),
                        trace_writer,
                    )
                    last_failure = str(exc)
                    step_failed = True
                    break
            elif step.tool == "ring_type_calculator":
                if not args.get("output_path"):
                    base_dir = context.get("output_dir") or context.get("cif_dir")
                    if base_dir:
                        args["output_path"] = str(Path(base_dir) / "ring_type_results.json")
                requested_max = args.get("max_ring", 12)
                try:
                    requested_max = int(requested_max)
                except (TypeError, ValueError):
                    requested_max = 12
                args["max_ring"] = requested_max
                if not args.get("preferred_validation"):
                    preferred = _preferred_ring_validation(request.user_request)
                    if preferred:
                        args["preferred_validation"] = preferred
                req = ring_tools.RingTypeRequest(**args)
                input_payload = req.model_dump()
                try:
                    _emit_trace_event(
                        trace_writer,
                        {
                            "tool": step.tool,
                            "status": "running",
                            "input": input_payload,
                            "loop": loop_count,
                        },
                    )
                    result = ring_tools.run_ring_type_calculator(req)
                    reference_cif_dir = context.get("reference_cif_dir")
                    if reference_cif_dir:
                        try:
                            if Path(req.cif_dir).resolve() == Path(reference_cif_dir).resolve():
                                context["reference_ring_info"] = result.model_dump().get("entries", [])
                        except Exception:
                            pass
                    for entry in result.entries:
                        if entry.cif_path in candidates:
                            cand = candidates[entry.cif_path]
                            cand.ring_types = entry.ring_types
                            cand.max_ring = entry.max_ring
                    _append_trace(
                        traces,
                        ToolTrace(
                            tool=step.tool,
                            status="success",
                            input=input_payload,
                            output=result.model_dump(),
                            loop=loop_count,
                        ),
                        trace_writer,
                    )
                except Exception as exc:
                    _append_trace(
                        traces,
                        ToolTrace(
                            tool=step.tool,
                            status="error",
                            input=input_payload,
                            output=None,
                            error=str(exc),
                            loop=loop_count,
                        ),
                        trace_writer,
                    )
                    last_failure = str(exc)
                    step_failed = True
                    break
            elif step.tool == "ring_size_calculator":
                req = ring_tools.RingSizeRequest(**args)
                context["last_ring_target"] = req.target_ring_size
                try:
                    _emit_trace_event(
                        trace_writer,
                        {
                            "tool": step.tool,
                            "status": "running",
                            "input": req.model_dump(),
                            "loop": loop_count,
                        },
                    )
                    result = ring_tools.run_ring_size_calculator(req)
                    if context.get("reference_cif_dir"):
                        try:
                            if Path(req.cif_dir).resolve() == Path(context["reference_cif_dir"]).resolve():
                                context["reference_ring_metrics"] = result.model_dump().get("entries", [])
                        except Exception:
                            pass
                    for entry in result.entries:
                        if entry.cif_path in candidates:
                            candidates[entry.cif_path].ring_metrics = entry.ring_sizes
                    _append_trace(
                        traces,
                        ToolTrace(
                            tool=step.tool,
                            status="success",
                            input=req.model_dump(),
                            output=result.model_dump(),
                            loop=loop_count,
                        ),
                        trace_writer,
                    )
                except Exception as exc:
                    _append_trace(
                        traces,
                        ToolTrace(
                            tool=step.tool,
                            status="error",
                            input=req.model_dump(),
                            output=None,
                            error=str(exc),
                            loop=loop_count,
                        ),
                        trace_writer,
                    )
                    last_failure = str(exc)
                    step_failed = True
                    break
            elif step.tool == "gulp_opt":
                req = gulp_opt.GulpOptimizationRequest(**args)
                try:
                    _emit_trace_event(
                        trace_writer,
                        {
                            "tool": step.tool,
                            "status": "running",
                            "input": req.model_dump(),
                            "loop": loop_count,
                        },
                    )
                    result = gulp_opt.run_gulp_optimization(req)
                    if (
                        args.get("framework") == context.get("reference_framework")
                        or args.get("cif_path") == context.get("reference_cif_path")
                    ):
                        context["reference_cif_path"] = result.optimized_cif_path
                        context["reference_cif_dir"] = str(
                            _single_cif_dir(Path(result.optimized_cif_path), "generation-ref-opt")
                        )
                        context["reference_optimized"] = True
                    _append_trace(
                        traces,
                        ToolTrace(
                            tool=step.tool,
                            status="success",
                            input=req.model_dump(),
                            output=result.model_dump(),
                            loop=loop_count,
                        ),
                        trace_writer,
                    )
                except Exception as exc:
                    _append_trace(
                        traces,
                        ToolTrace(
                            tool=step.tool,
                            status="error",
                            input=req.model_dump(),
                            output=None,
                            error=str(exc),
                            loop=loop_count,
                        ),
                        trace_writer,
                    )
                    last_failure = str(exc)
                    step_failed = True
                    break
            elif step.tool == "zeopp_batch":
                if "cif_paths" not in args or not args.get("cif_paths"):
                    args["cif_paths"] = list(candidates.keys())
                elif isinstance(args.get("cif_paths"), list):
                    placeholder_tokens = {"{candidate_cif_paths}", "{cif_paths}", "{candidate_paths}"}
                    if any(str(item) in placeholder_tokens or ("{" in str(item) and "}" in str(item)) for item in args["cif_paths"]):
                        args["cif_paths"] = list(candidates.keys())
                elif isinstance(args.get("cif_paths"), str):
                    if "{" in args["cif_paths"] and "}" in args["cif_paths"]:
                        args["cif_paths"] = list(candidates.keys())
                calc_types = args.get("calculation_types")
                if isinstance(calc_types, list):
                    context["last_zeopp_calculation_types"] = list(calc_types)
                if not _requires_channel_analysis(request.user_request):
                    if isinstance(calc_types, list) and "channel_analysis" in calc_types:
                        args["calculation_types"] = [t for t in calc_types if t != "channel_analysis"]
                        if not args["calculation_types"]:
                            args["calculation_types"] = ["pore_diameters"]
                        args.pop("channel_probe_radius", None)
                req = zeopp.ZeoppBatchRequest(**args)
                try:
                    _emit_trace_event(
                        trace_writer,
                        {
                            "tool": step.tool,
                            "status": "running",
                            "input": req.model_dump(),
                            "loop": loop_count,
                        },
                    )
                    result = zeopp.run_zeopp_batch(req)
                    for entry in result.entries:
                        if entry.cif_path in candidates:
                            candidates[entry.cif_path].zeopp = {
                                "results": {
                                    key: _safe_model_dump(value, exclude={"raw_output"})
                                    for key, value in entry.results.items()
                                },
                                "errors": entry.errors,
                            }
                    _append_trace(
                        traces,
                        ToolTrace(
                            tool=step.tool,
                            status="success",
                            input=req.model_dump(),
                            output=result.model_dump(),
                            loop=loop_count,
                        ),
                        trace_writer,
                    )
                except Exception as exc:
                    _append_trace(
                        traces,
                        ToolTrace(
                            tool=step.tool,
                            status="error",
                            input=req.model_dump(),
                            output=None,
                            error=str(exc),
                            loop=loop_count,
                        ),
                        trace_writer,
                    )
                    last_failure = str(exc)
                    step_failed = True
                    break
            elif step.tool == "iza_match":
                if not args.get("output_path"):
                    base_dir = context.get("output_dir") or context.get("cif_dir")
                    if base_dir:
                        args["output_path"] = str(Path(base_dir) / "iza_match_results.json")
                if args.get("iza_dir"):
                    iza_candidate = get_settings().resolve_path(Path(str(args["iza_dir"])))
                    if iza_candidate.exists():
                        args["iza_dir"] = str(iza_candidate)
                    else:
                        args.pop("iza_dir", None)
                req = iza_match.IzaMatchRequest(**args)
                try:
                    _emit_trace_event(
                        trace_writer,
                        {
                            "tool": step.tool,
                            "status": "running",
                            "input": req.model_dump(),
                            "loop": loop_count,
                        },
                    )
                    result = iza_match.run_iza_match(req)
                    for entry in result.entries:
                        if entry.cif_path in candidates:
                            cand = candidates[entry.cif_path]
                            cand.iza_code = entry.iza_code
                            cand.iza_status = entry.status
                    _append_trace(
                        traces,
                        ToolTrace(
                            tool=step.tool,
                            status="success",
                            input=req.model_dump(),
                            output=result.model_dump(),
                            loop=loop_count,
                        ),
                        trace_writer,
                    )
                except Exception as exc:
                    _append_trace(
                        traces,
                        ToolTrace(
                            tool=step.tool,
                            status="error",
                            input=req.model_dump(),
                            output=None,
                            error=str(exc),
                            loop=loop_count,
                        ),
                        trace_writer,
                    )
                    last_failure = str(exc)
                    step_failed = True
                    break

        if step_failed:
            loop_count += 1
            continue

        if candidates:
            separation_task = bool(separation_context.get("is_separation_task"))
            if separation_task:
                missing_lfs = [
                    cand.cif_path
                    for cand in candidates.values()
                    if _candidate_largest_free_sphere(cand) is None
                ]
                if missing_lfs:
                    try:
                        req = zeopp.ZeoppBatchRequest(
                            cif_paths=list(candidates.keys()),
                            calculation_types=["pore_diameters"],
                        )
                        _emit_trace_event(
                            trace_writer,
                            {
                                "tool": "zeopp_batch",
                                "status": "running",
                                "input": req.model_dump(),
                                "loop": loop_count,
                            },
                        )
                        result = zeopp.run_zeopp_batch(req)
                        for entry in result.entries:
                            if entry.cif_path in candidates:
                                candidates[entry.cif_path].zeopp = {
                                    "results": {
                                        key: _safe_model_dump(value, exclude={"raw_output"})
                                        for key, value in entry.results.items()
                                    },
                                    "errors": entry.errors,
                                }
                        _append_trace(
                            traces,
                            ToolTrace(
                                tool="zeopp_batch",
                                status="success",
                                input=req.model_dump(),
                                output=result.model_dump(),
                                loop=loop_count,
                            ),
                            trace_writer,
                        )
                    except Exception as exc:
                        _append_trace(
                            traces,
                            ToolTrace(
                                tool="zeopp_batch",
                                status="error",
                                input={"cif_paths": list(candidates.keys()), "calculation_types": ["pore_diameters"]},
                                output=None,
                                error=str(exc),
                                loop=loop_count,
                            ),
                            trace_writer,
                        )
                        last_failure = f"Separation screening failed to compute pore diameters: {exc}"
                        loop_count += 1
                        continue

            candidate_summary = _summarize_generation_candidates(candidates)
            reference_ring_info = context.get("reference_ring_info") or []
            novelty_required = _requires_novelty(request.user_request)
            excluded_frameworks = _normalize_frameworks(request.excluded_frameworks or [])

            need_ring_metrics = any(cand.ring_metrics for cand in candidates.values()) or context.get("last_ring_target")
            need_zeopp = any(cand.zeopp for cand in candidates.values()) or context.get("last_zeopp_calculation_types")
            reference_cif_path = context.get("reference_cif_path")

            if (need_ring_metrics or need_zeopp) and reference_cif_path:
                if not context.get("reference_optimized"):
                    try:
                        req = gulp_opt.GulpOptimizationRequest(cif_path=reference_cif_path)
                        _emit_trace_event(
                            trace_writer,
                            {
                                "tool": "gulp_opt",
                                "status": "running",
                                "input": req.model_dump(),
                                "loop": loop_count,
                            },
                        )
                        result = gulp_opt.run_gulp_optimization(req)
                        context["reference_cif_path"] = result.optimized_cif_path
                        context["reference_cif_dir"] = str(
                            _single_cif_dir(Path(result.optimized_cif_path), "generation-ref-opt")
                        )
                        context["reference_optimized"] = True
                        _append_trace(
                            traces,
                            ToolTrace(
                                tool="gulp_opt",
                                status="success",
                                input=req.model_dump(),
                                output=result.model_dump(),
                                loop=loop_count,
                            ),
                            trace_writer,
                        )
                    except Exception as exc:
                        _append_trace(
                            traces,
                            ToolTrace(
                                tool="gulp_opt",
                                status="error",
                                input={"cif_path": reference_cif_path},
                                output=None,
                                error=str(exc),
                                loop=loop_count,
                            ),
                            trace_writer,
                        )
                if need_ring_metrics and not context.get("reference_ring_metrics"):
                    ref_dir = context.get("reference_cif_dir")
                    if not ref_dir and context.get("reference_cif_path"):
                        ref_dir = str(
                            _single_cif_dir(Path(context["reference_cif_path"]), "generation-ref")
                        )
                        context["reference_cif_dir"] = ref_dir
                    if ref_dir:
                        target_ring = context.get("last_ring_target") or 8
                        try:
                            req = ring_tools.RingSizeRequest(
                                cif_dir=ref_dir, target_ring_size=target_ring
                            )
                            _emit_trace_event(
                                trace_writer,
                                {
                                    "tool": "ring_size_calculator",
                                    "status": "running",
                                    "input": req.model_dump(),
                                    "loop": loop_count,
                                },
                            )
                            result = ring_tools.run_ring_size_calculator(req)
                            context["reference_ring_metrics"] = result.model_dump().get("entries", [])
                            _append_trace(
                                traces,
                                ToolTrace(
                                    tool="ring_size_calculator",
                                    status="success",
                                    input=req.model_dump(),
                                    output=result.model_dump(),
                                    loop=loop_count,
                                ),
                                trace_writer,
                            )
                        except Exception as exc:
                            _append_trace(
                                traces,
                                ToolTrace(
                                    tool="ring_size_calculator",
                                    status="error",
                                    input={"cif_dir": ref_dir, "target_ring_size": target_ring},
                                    output=None,
                                    error=str(exc),
                                    loop=loop_count,
                                ),
                                trace_writer,
                            )
                if need_zeopp and not context.get("reference_zeopp") and context.get("reference_cif_path"):
                    calc_types = context.get("last_zeopp_calculation_types") or ["pore_diameters"]
                    if not _requires_channel_analysis(request.user_request):
                        calc_types = [t for t in calc_types if t != "channel_analysis"]
                        if not calc_types:
                            calc_types = ["pore_diameters"]
                    try:
                        req = zeopp.ZeoppBatchRequest(
                            cif_paths=[context["reference_cif_path"]],
                            calculation_types=calc_types,
                        )
                        _emit_trace_event(
                            trace_writer,
                            {
                                "tool": "zeopp_batch",
                                "status": "running",
                                "input": req.model_dump(),
                                "loop": loop_count,
                            },
                        )
                        result = zeopp.run_zeopp_batch(req)
                        if result.entries:
                            context["reference_zeopp"] = _safe_model_dump(
                                result.entries[0], exclude={"raw_output"}
                            )
                        _append_trace(
                            traces,
                            ToolTrace(
                                tool="zeopp_batch",
                                status="success",
                                input=req.model_dump(),
                                output=result.model_dump(),
                                loop=loop_count,
                            ),
                            trace_writer,
                        )
                    except Exception as exc:
                        _append_trace(
                            traces,
                            ToolTrace(
                                tool="zeopp_batch",
                                status="error",
                                input={"cif_paths": [context.get("reference_cif_path")], "calculation_types": calc_types},
                                output=None,
                                error=str(exc),
                                loop=loop_count,
                            ),
                            trace_writer,
                        )
            if excluded_frameworks:
                missing_match_info = any(
                    cand.iza_code is None and cand.iza_status is None for cand in candidates.values()
                )
                if missing_match_info:
                    base_dir = context.get("output_dir") or context.get("cif_dir")
                    if base_dir:
                        try:
                            req = iza_match.IzaMatchRequest(
                                cif_dir=base_dir,
                                output_path=str(Path(base_dir) / "iza_match_results.json"),
                            )
                            _emit_trace_event(
                                trace_writer,
                                {
                                    "tool": "iza_match",
                                    "status": "running",
                                    "input": req.model_dump(),
                                    "loop": loop_count,
                                },
                            )
                            result = iza_match.run_iza_match(req)
                            for entry in result.entries:
                                if entry.cif_path in candidates:
                                    cand = candidates[entry.cif_path]
                                    cand.iza_code = entry.iza_code
                                    cand.iza_status = entry.status
                            _append_trace(
                                traces,
                                ToolTrace(
                                    tool="iza_match",
                                    status="success",
                                    input=req.model_dump(),
                                    output=result.model_dump(),
                                    loop=loop_count,
                                ),
                                trace_writer,
                            )
                        except Exception as exc:
                            _append_trace(
                                traces,
                                ToolTrace(
                                    tool="iza_match",
                                    status="error",
                                    input={"cif_dir": base_dir},
                                    output=None,
                                    error=str(exc),
                                    loop=loop_count,
                                ),
                                trace_writer,
                            )
                excluded_set = {fw.upper() for fw in excluded_frameworks}
                candidates = {
                    path: cand
                    for path, cand in candidates.items()
                    if not (cand.iza_code and cand.iza_code.upper() in excluded_set)
                }
                candidate_summary = _summarize_generation_candidates(candidates)
                if not candidates:
                    last_failure = "All candidates excluded by framework constraints."
                    loop_count += 1
                    continue

            # Hard constraint: zeolites with max_ring < 8 are never eligible.
            min_allowed_max_ring = 8
            removed_by_ring = [
                cand.cif_path for cand in candidates.values() if int(cand.max_ring or 0) < min_allowed_max_ring
            ]
            if removed_by_ring:
                candidates = {
                    path: cand
                    for path, cand in candidates.items()
                    if int(cand.max_ring or 0) >= min_allowed_max_ring
                }
                _append_trace(
                    traces,
                    ToolTrace(
                        tool="hard_filter_max_ring",
                        status="info",
                        input={"min_allowed_max_ring": min_allowed_max_ring},
                        output={"removed_cifs": removed_by_ring, "remaining": len(candidates)},
                        loop=loop_count,
                    ),
                    trace_writer,
                )
            if not candidates:
                last_failure = f"All candidates removed by hard constraint: max_ring >= {min_allowed_max_ring}."
                loop_count += 1
                continue

            if separation_task:
                permeate_species = separation_context.get("permeate_species")
                permeate_diameter = separation_context.get("permeate_diameter_A")
                if not permeate_species or permeate_diameter is None:
                    _append_trace(
                        traces,
                        ToolTrace(
                            tool="hard_filter_separation_lfs",
                            status="error",
                            input=_public_separation_context(separation_context),
                            output={"removed_cifs": list(candidates.keys()), "remaining": 0},
                            error="Missing permeate species or kinetic diameter for separation filtering.",
                            loop=loop_count,
                        ),
                        trace_writer,
                    )
                    last_failure = "Unable to resolve permeate species/diameter for separation filtering."
                    loop_count += 1
                    continue

                removed_by_lfs: List[Dict[str, Any]] = []
                kept: Dict[str, GenerationCandidate] = {}
                for path, cand in candidates.items():
                    lfs = _candidate_largest_free_sphere(cand)
                    if lfs is None or lfs < float(permeate_diameter):
                        removed_by_lfs.append(
                            {
                                "cif_path": path,
                                "largest_free_sphere": lfs,
                            }
                        )
                        continue
                    kept[path] = cand
                candidates = kept
                _append_trace(
                    traces,
                    ToolTrace(
                        tool="hard_filter_separation_lfs",
                        status="info",
                        input={
                            "permeate_species": permeate_species,
                            "permeate_diameter_A": float(permeate_diameter),
                        },
                        output={
                            "removed": removed_by_lfs,
                            "remaining": len(candidates),
                        },
                        loop=loop_count,
                    ),
                    trace_writer,
                )
                if not candidates:
                    last_failure = (
                        f"All candidates removed by separation passability constraint: "
                        f"largest_free_sphere >= {float(permeate_diameter):.2f} A for {permeate_species}."
                    )
                    loop_count += 1
                    continue

            candidate_summary = _summarize_generation_candidates(candidates)
            critic_prompt = _generation_critic_prompt()
            critic_raw = _invoke_generation_critic_llm(
                llm,
                critic_prompt,
                {
                    "user_request": request.user_request,
                    "novelty_required": novelty_required,
                    "reference_reason": context.get("reference_reason") or "",
                    "reference_ring_info": json.dumps(reference_ring_info, ensure_ascii=False),
                    "reference_ring_metrics": json.dumps(context.get("reference_ring_metrics") or [], ensure_ascii=False),
                    "reference_zeopp": json.dumps(context.get("reference_zeopp") or {}, ensure_ascii=False),
                    "plan": json.dumps(plan.model_dump() if plan is not None else {}, ensure_ascii=False),
                    "loop_count": loop_count,
                    "max_loops": max_loops,
                    "candidates": json.dumps(candidate_summary, ensure_ascii=False),
                    "separation_context": json.dumps(
                        _critic_separation_context(separation_context), ensure_ascii=False
                    ),
                },
            )
            critic_decision, critic_reason, selected_cifs, top_k = _parse_generation_critic(critic_raw)
            if not novelty_required and critic_decision == "replan":
                reason_lower = (critic_reason or "").lower()
                if any(token in reason_lower for token in ("iza", "novel", "hypothetical", "not in database")):
                    critic_decision = "accept"
                    critic_reason = "Novelty not required; accepting best available candidates."
            _append_trace(
                traces,
                    ToolTrace(
                        tool="generation_critic",
                        status="success",
                        input={
                            "candidates": candidate_summary,
                            "reference_reason": context.get("reference_reason") or "",
                            "reference_ring_info": reference_ring_info,
                            "reference_ring_metrics": context.get("reference_ring_metrics") or [],
                            "reference_zeopp": context.get("reference_zeopp") or {},
                            "novelty_required": novelty_required,
                            "separation_context": _critic_separation_context(separation_context),
                        },
                    output={
                        "decision": critic_decision,
                        "reason": critic_reason,
                        "selected_cifs": selected_cifs,
                        "top_k": top_k,
                        "raw": critic_raw,
                    },
                    loop=loop_count,
                ),
                trace_writer,
            )
            if critic_decision == "replan":
                if loop_count + 1 >= max_loops:
                    last_failure = "No suitable new framework found that satisfies semantic criteria."
                    loop_count += 1
                    break
                last_failure = critic_reason or "Semantic critic requested replan"
                loop_count += 1
                continue
            selected_candidates = _select_candidates(candidates, selected_cifs, top_k)
            return GenerationResult(
                status="success",
                candidates=selected_candidates,
                output_dir=output_dir,
                loop_count=loop_count,
                traces=[t.as_dict() for t in traces],
                reference_lookup=reference_lookup_data,
            )

        last_failure = "No candidates generated in this loop"
        loop_count += 1

    return GenerationResult(
        status="failed",
        candidates=[],
        output_dir=output_dir,
        loop_count=loop_count,
        failure_reason=last_failure,
        traces=[t.as_dict() for t in traces],
        reference_lookup=reference_lookup_data,
    )

def _finalize_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Messages: {messages}\nMemory: {memory}\nPlan: {plan}\nTraces: {traces}\nDraft: {draft}\n"
                "Use only tool outputs from traces. If pore_diameters are available, list all reported metrics "
                "and their values (largest included sphere, largest free sphere, and largest included sphere along free sphere path) "
                "with units, except when answering diffusion why/factors questions that already have ring_size_calculator evidence. For diffusion why/factors questions, follow the ML importance framing but focus on ring size: when ring_size_calculator results exist for the compared structures, explain differences using ring size evidence only (e.g., 8-ring dmin/dmax comparisons), and do not introduce other drivers. Otherwise answer concisely. "
                "Return plain text.",
            ),
        ]
    )


def build_agent_graph(
    llm: BaseLanguageModel,
    search_backend: Optional[Any] = None,
    trace_writer: TraceWriter = None,
    generation_llm: Optional[BaseLanguageModel] = None,
):
    backend = search_backend
    generation_llm_instance = generation_llm
    graph = StateGraph(GraphState)

    def _get_generation_llm() -> BaseLanguageModel:
        nonlocal generation_llm_instance
        if generation_llm_instance is None:
            generation_llm_instance = build_generation_llm()
        return generation_llm_instance

    def classify_intent(state: GraphState) -> GraphState:
        user_text = _latest_user_text(state["messages"])
        _resolve_framework(user_text, state["memory"])
        _update_memory_frameworks(user_text, state["memory"])
        _extract_temp_loading(user_text, state["memory"])
        is_param_update = _is_parameter_update(user_text)
        if not is_param_update:
            state["memory"]["last_user_request"] = user_text
        intent, reason = _classify_intent(
            llm,
            user_text,
            state["memory"],
            _serialize_messages(state["messages"]),
        )
        if is_param_update and intent == "generate":
            intent = "general"
            reason = "Parameter update detected; treat as general."
        state["intent"] = intent
        state["intent_reason"] = reason
        state["reasoning"].append(f"Intent: {intent} ({reason})")
        return state

    def plan(state: GraphState) -> GraphState:
        user_text = _latest_user_text(state["messages"])
        request_text = user_text
        if _is_parameter_update(user_text):
            prior_request = state["memory"].get("last_user_request")
            if prior_request:
                request_text = prior_request

        def _generation_tool(request: GenerationSubgraphRequest) -> GenerationResult:
            return run_generation_subgraph(_get_generation_llm(), request, trace_writer=trace_writer)

        tool_specs: Dict[str, ToolSpec] = {
            "diffusion_predictor": ToolSpec(
                name="diffusion_predictor",
                input_model=DiffusionRequest,
                handler=_tool_diffusion,
                description=(
                    "Predict ethene diffusion coefficients (m^2/s). "
                    "Use for single-framework estimates or comparisons. "
                    "Args: framework, temperature_K, loading_per_uc."
                ),
            ),
            "zeopp": ToolSpec(
                name="zeopp",
                input_model=zeopp.ZeoppRequest,
                handler=_tool_zeopp,
                description="Run zeo++ for pore_diameters/surface_area/volume. Args: framework, calculation_type, probe_radius.",
            ),
            "gulp_opt": ToolSpec(
                name="gulp_opt",
                input_model=gulp_opt.GulpOptimizationRequest,
                handler=lambda req: gulp_opt.run_gulp_optimization(req),
                description="Optimize a CIF with GULP using zeolite force-field defaults. Args: framework or cif_path.",
            ),
            "ring_type_calculator": ToolSpec(
                name="ring_type_calculator",
                input_model=ring_tools.RingTypeRequest,
                handler=lambda req: ring_tools.run_ring_type_calculator(req),
                description="Compute ring types/counts for CIFs. Args: cif_dir, max_ring.",
            ),
            "ring_size_calculator": ToolSpec(
                name="ring_size_calculator",
                input_model=ring_tools.RingSizeRequest,
                handler=lambda req: ring_tools.run_ring_size_calculator(req),
                description=(
                    "Compute ring aperture sizes (effective diameters) for a target ring (default 8). "
                    "Args: cif_dir, target_ring_size, frameworks. "
                    "Require frameworks (list of IZA codes) so a dedicated CIF directory is built and the full library is never scanned. "
                    "Example: for 'EAB vs CHA' comparisons, frameworks: [\"EAB\", \"CHA\"]. "
                    "Framework CIFs resolve under data/cif_files by default. "
                    "For diffusion explanation requests, prefer this tool as the structural explainer. "
                    "Do NOT analyze diffusion reasons unless the user explicitly asks why/factors."
                ),
            ),
            "iza_match": ToolSpec(
                name="iza_match",
                input_model=iza_match.IzaMatchRequest,
                handler=lambda req: iza_match.run_iza_match(req),
                description=(
                    "Check whether CIFs match the IZA database. "
                    "Use when the user asks whether a structure exists in the IZA database. Args: cif_dir."
                ),
            ),
            "generation_subgraph": ToolSpec(
                name="generation_subgraph",
                input_model=GenerationSubgraphRequest,
                handler=_generation_tool,
                description="Run generation subgraph. Args: user_request.",
            ),
        }
        system_prompt = (
            "You are ZeoAgent's Orchestrator Planner for zeolite tasks. Output JSON with keys: goal, steps. "
            "Each step must include tool, args, expected_outputs, success_criteria. Use only listed tools. "
            "If intent is generate, include a single generation_subgraph step. "
            "If intent is not generate, do NOT use generation_subgraph. "
            "For diffusion coefficient questions, use diffusion_predictor; "
            "do NOT use ring_size_calculator unless the user explicitly asks why/factors. "
            "For zeopp surface_area, volume, or channel_analysis in general questions, "
            "set probe_radius to 1.5 unless the user specifies a different value. "
            "Do not include a dissatisfied field for generation_subgraph; it is inferred by the executor."
        )
        plan_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                (
                    "human",
                    "User request: {user_request}\nIntent: {intent}\nMemory: {memory}\nMessages: {messages}\nArgs format: list of {key, value} objects.\nTools:\n{tools}\nReturn JSON only.",
                ),
            ]
        )
        tools_prompt = _build_tool_prompt(tool_specs)
        memory_json = json.dumps(state["memory"], ensure_ascii=False)
        raw_plan = ""
        plan_obj = None
        plan_error = None
        missing_fields: List[str] = []
        try:
            client = _build_general_responses_client()
            model = (get_settings().generation_model or "gpt-5").strip()
            plan_obj, raw_plan, plan_error, missing_fields = _run_general_plan_responses(
                client,
                model,
                system_prompt,
                request_text,
                state["intent"],
                memory_json,
                tools_prompt,
                tool_specs,
                json.dumps(_serialize_messages(state["messages"]), ensure_ascii=False),
            )
        except Exception as exc:
            plan_obj = None
            plan_error = plan_error or f"Responses planning failed: {exc}"
        if plan_obj is None:
            plan_error = plan_error or "Responses planning failed"
        if plan_obj is not None:
            _inject_ring_size_frameworks(plan_obj, request_text, state["memory"])
        state["plan"] = plan_obj.model_dump() if plan_obj else None
        state["plan_obj"] = plan_obj
        state["plan_raw"] = raw_plan
        state["plan_error"] = plan_error
        state["missing_fields"] = missing_fields
        state["reasoning"].append(f"Plan error: {plan_error}" if plan_error else "Plan OK")
        return state

    def execute(state: GraphState) -> GraphState:
        plan_obj = state.get("plan_obj")
        state["tool_payload"] = {}
        state["last_tool"] = None
        state["missing_inputs"] = []
        if state.get("plan_error") or plan_obj is None:
            error_msg = state.get("plan_error") or "Plan missing"
            _append_trace(
                state["traces"],
                ToolTrace(
                    tool="plan_validation",
                    status="error",
                    input={},
                    output=None,
                    error=error_msg,
                    loop=state["loop_count"],
                ),
                trace_writer,
            )
            return state

        tool_specs: Dict[str, ToolSpec] = {
            "diffusion_predictor": ToolSpec(
                name="diffusion_predictor",
                input_model=DiffusionRequest,
                handler=_tool_diffusion,
                description=(
                    "Predict ethene diffusion coefficients (m^2/s). "
                    "Use for single-framework estimates or comparisons. "
                    "Args: framework, temperature_K, loading_per_uc."
                ),
            ),
            "zeopp": ToolSpec(
                name="zeopp",
                input_model=zeopp.ZeoppRequest,
                handler=_tool_zeopp,
                description="Run zeo++ for pore_diameters/surface_area/volume.",
            ),
            "gulp_opt": ToolSpec(
                name="gulp_opt",
                input_model=gulp_opt.GulpOptimizationRequest,
                handler=lambda req: gulp_opt.run_gulp_optimization(req),
                description="Optimize a CIF with GULP using zeolite force-field defaults.",
            ),
            "ring_type_calculator": ToolSpec(
                name="ring_type_calculator",
                input_model=ring_tools.RingTypeRequest,
                handler=lambda req: ring_tools.run_ring_type_calculator(req),
                description="Compute ring types/counts for CIFs.",
            ),
            "ring_size_calculator": ToolSpec(
                name="ring_size_calculator",
                input_model=ring_tools.RingSizeRequest,
                handler=lambda req: ring_tools.run_ring_size_calculator(req),
                description="Compute ring aperture sizes (effective diameters) for a target ring (default 8). Args: cif_dir, target_ring_size. Framework CIFs resolve under data/cif_files by default; if a prior gulp_opt produced an optimized CIF, prefer that. For diffusion why/factors questions, ML importance indicates ring size is a dominant driver, so prefer this tool as the structural explainer. Do NOT analyze diffusion reasons unless the user explicitly asks why/factors.",
            ),
            "iza_match": ToolSpec(
                name="iza_match",
                input_model=iza_match.IzaMatchRequest,
                handler=lambda req: iza_match.run_iza_match(req),
                description="Check whether CIFs match the IZA database.",
            ),
            "generation_subgraph": ToolSpec(
                name="generation_subgraph",
                input_model=GenerationSubgraphRequest,
                handler=lambda req: run_generation_subgraph(_get_generation_llm(), req, trace_writer=trace_writer),
                description="Run generation subgraph.",
            ),
        }
        for step in plan_obj.steps:
            spec = tool_specs[step.tool]
            args = dict(step.args)
            if step.tool == "diffusion_predictor":
                args.setdefault("framework", state["memory"].get("framework"))
                user_temp = state["memory"].get("temperature_K")
                user_loading = state["memory"].get("loading_per_uc")
                if user_temp is not None:
                    args["temperature_K"] = user_temp
                else:
                    args.pop("temperature_K", None)
                if user_loading is not None:
                    args["loading_per_uc"] = user_loading
                else:
                    args.pop("loading_per_uc", None)
                missing = []
                if user_temp is None:
                    missing.append("temperature (K)")
                if user_loading is None:
                    missing.append("loading (molecules per unit cell)")
                if missing:
                    state["missing_inputs"] = missing
                    _append_trace(
                        state["traces"],
                        ToolTrace(
                            tool=step.tool,
                            status="needs_input",
                            input=args,
                            output=None,
                            error=f"Missing required inputs: {', '.join(missing)}",
                            loop=state["loop_count"],
                        ),
                        trace_writer,
                    )
                    break
            if step.tool == "zeopp":
                args.setdefault("framework", state["memory"].get("framework"))
            if step.tool == "gulp_opt":
                args.setdefault("framework", state["memory"].get("framework"))
            if step.tool == "ring_type_calculator":
                cif_dir_val = args.get("cif_dir")
                if cif_dir_val:
                    try:
                        cif_dir_path = Path(str(cif_dir_val))
                        settings = get_settings()
                        cif_root = settings.resolve_path(settings.data_root) / "cif_files"
                        if cif_dir_path.resolve() == cif_root.resolve():
                            # Avoid scanning the full CIF library when a specific framework is known.
                            args.pop("cif_dir", None)
                    except Exception:
                        pass
                if not args.get("cif_dir"):
                    fw = args.get("framework") or state["memory"].get("framework")
                    if fw:
                        args["cif_dir"] = str(resolve_cif_path(fw))
            if step.tool == "ring_size_calculator":
                last_opt = (state["memory"].get("last_gulp_opt") or {}).get("optimized_cif_path")
                last_opt_path = Path(last_opt) if last_opt else None
                cif_dir_val = args.get("cif_dir")
                frameworks = args.get("frameworks")
                frameworks_provided = frameworks is not None
                if frameworks is None or (isinstance(frameworks, list) and not frameworks):
                    pool: List[str] = []
                    mem_frameworks = state["memory"].get("frameworks")
                    if isinstance(mem_frameworks, list):
                        pool.extend(mem_frameworks)
                    mem_framework = state["memory"].get("framework")
                    if mem_framework:
                        pool.append(mem_framework)
                    framework_arg = args.get("framework")
                    if framework_arg:
                        pool.append(framework_arg)
                    pool = _normalize_frameworks(pool)
                    if pool:
                        args["frameworks"] = pool
                        frameworks = pool
                        frameworks_provided = True
                if frameworks is not None:
                    if not isinstance(frameworks, list) or not frameworks:
                        state["missing_inputs"] = ["frameworks"]
                        _append_trace(
                            state["traces"],
                            ToolTrace(
                                tool=step.tool,
                                status="needs_input",
                                input=args,
                                output=None,
                                error="Missing required inputs: frameworks",
                                loop=state["loop_count"],
                            ),
                            trace_writer,
                        )
                        break
                    frameworks = [str(item).strip().upper() for item in frameworks if str(item).strip()]
                    if not frameworks:
                        state["missing_inputs"] = ["frameworks"]
                        _append_trace(
                            state["traces"],
                            ToolTrace(
                                tool=step.tool,
                                status="needs_input",
                                input=args,
                                output=None,
                                error="Missing required inputs: frameworks",
                                loop=state["loop_count"],
                            ),
                            trace_writer,
                        )
                        break
                    args["cif_dir"] = str(_multi_cif_dir(frameworks, "ring-size"))
                    args.pop("frameworks", None)
                    cif_dir_val = args.get("cif_dir")
                    cif_dir_path = Path(str(cif_dir_val)) if cif_dir_val else None
                else:
                    cif_dir_path = Path(str(cif_dir_val)) if cif_dir_val else None
                if cif_dir_path:
                    try:
                        settings = get_settings()
                        cif_root = settings.resolve_path(settings.data_root) / "cif_files"
                        if cif_dir_path.resolve() == cif_root.resolve():
                            # Avoid scanning the full CIF library when a specific framework is known.
                            args.pop("cif_dir", None)
                            cif_dir_val = None
                            cif_dir_path = None
                    except Exception:
                        pass

                use_last_opt = False
                if last_opt_path and last_opt_path.exists():
                    if frameworks_provided:
                        use_last_opt = False
                    elif not cif_dir_val:
                        use_last_opt = True
                    elif cif_dir_path and not cif_dir_path.exists():
                        # If planner passed a framework token (e.g., "EAB"), prefer the optimized CIF.
                        fw_token = str(cif_dir_val).upper()
                        mem_fw = str(state["memory"].get("framework") or "").upper()
                        arg_fw = str(args.get("framework") or "").upper()
                        if fw_token == mem_fw or arg_fw == mem_fw or not arg_fw:
                            use_last_opt = True

                if use_last_opt:
                    args["cif_dir"] = str(_single_cif_dir(last_opt_path, "ring-size"))
                else:
                    if cif_dir_path and not cif_dir_path.exists():
                        # Planner may pass a framework code (e.g., "EAB") instead of a directory.
                        fw_candidate = args.get("framework") or str(cif_dir_val)
                        resolved_dir = _resolve_cif_dir_for_tool(
                            None,
                            fw_candidate,
                            state["memory"].get("framework"),
                            "ring-size",
                        )
                    else:
                        resolved_dir = _resolve_cif_dir_for_tool(
                            cif_dir_val,
                            args.get("framework"),
                            state["memory"].get("framework"),
                            "ring-size",
                        )
                    if resolved_dir:
                        args["cif_dir"] = resolved_dir
                    if not args.get("cif_dir"):
                        state["missing_inputs"] = ["framework"]
                        _append_trace(
                            state["traces"],
                            ToolTrace(
                                tool=step.tool,
                                status="needs_input",
                                input=args,
                                output=None,
                                error="Missing required inputs: framework",
                                loop=state["loop_count"],
                            ),
                            trace_writer,
                        )
                        break
            if step.tool == "iza_match":
                cif_dir_val = args.get("cif_dir")
                if cif_dir_val:
                    try:
                        cif_dir_path = Path(str(cif_dir_val))
                        settings = get_settings()
                        cif_root = settings.resolve_path(settings.data_root) / "cif_files"
                        if cif_dir_path.resolve() == cif_root.resolve():
                            # Avoid scanning the full CIF library when a specific framework is known.
                            args.pop("cif_dir", None)
                    except Exception:
                        pass
                resolved_dir = _resolve_cif_dir_for_tool(
                    args.get("cif_dir"),
                    args.get("framework"),
                    state["memory"].get("framework"),
                    "iza-match",
                )
                if resolved_dir:
                    args["cif_dir"] = resolved_dir
            if step.tool == "generation_subgraph":
                user_text = _latest_user_text(state["messages"])
                planned_request = args.get("user_request")
                if planned_request and planned_request != user_text:
                    args.setdefault("planner_guidance", planned_request)
                args["user_request"] = user_text
                args.setdefault("dissatisfied", _is_dissatisfied(user_text))
                args.setdefault("messages", _serialize_messages(state["messages"]))
                mem_frameworks = state["memory"].get("frameworks")
                if isinstance(mem_frameworks, list) and mem_frameworks:
                    args.setdefault("excluded_frameworks", _normalize_frameworks(mem_frameworks))
            _emit_trace_event(
                trace_writer,
                {
                    "tool": step.tool,
                    "status": "running",
                    "input": args,
                    "loop": state["loop_count"],
                },
            )
            try:
                req = spec.input_model(**args)
            except ValidationError as exc:
                _append_trace(
                    state["traces"],
                    ToolTrace(
                        tool=step.tool,
                        status="error",
                        input=step.args,
                        output=None,
                        error=f"Validation failed: {exc}",
                        loop=state["loop_count"],
                    ),
                    trace_writer,
                )
                break

            try:
                output = spec.handler(req)
                payload = _safe_model_dump(output, exclude={"raw_output"}) if output is not None else {}
                if step.tool == "generation_subgraph":
                    sub_traces = payload.get("traces", [])
                    for entry in sub_traces:
                        if not isinstance(entry, dict):
                            continue
                        entry_output = entry.get("output")
                        if not _keep_trace_raw():
                            entry_output = _strip_raw_fields(entry_output)
                        state["traces"].append(
                            ToolTrace(
                                tool=str(entry.get("tool", "generation_subgraph")),
                                status=str(entry.get("status", "info")),
                                input=entry.get("input") or {},
                                output=entry_output,
                                error=entry.get("error"),
                                loop=int(entry.get("loop", state["loop_count"])),
                            )
                        )
                trace = ToolTrace(
                    tool=step.tool,
                    status="success",
                    input=req.model_dump(),
                    output=payload,
                    loop=state["loop_count"],
                )
                _append_trace(state["traces"], trace, trace_writer)
                state["last_tool"] = step.tool
                state["tool_payload"] = payload
                if step.tool == "diffusion_predictor":
                    state["memory"]["framework"] = payload.get("framework")
                    if req.temperature_K is not None:
                        state["memory"]["temperature_K"] = req.temperature_K
                    if req.loading_per_uc is not None:
                        state["memory"]["loading_per_uc"] = req.loading_per_uc
                if step.tool == "zeopp":
                    state["memory"]["framework"] = payload.get("framework")
                if step.tool == "gulp_opt":
                    if req.framework:
                        state["memory"]["framework"] = req.framework
                    state["memory"]["last_gulp_opt"] = {
                        "input_cif_path": payload.get("input_cif_path"),
                        "optimized_cif_path": payload.get("optimized_cif_path"),
                        "run_dir": payload.get("run_dir"),
                    }
                if step.tool == "generation_subgraph":
                    state["memory"]["last_generation"] = _summarize_generation_result(payload)
            except Exception as exc:
                _append_trace(
                    state["traces"],
                    ToolTrace(
                        tool=step.tool,
                        status="error",
                        input=req.model_dump(),
                        output=None,
                        error=str(exc),
                        loop=state["loop_count"],
                    ),
                    trace_writer,
                )
                break

        return state

    def critic(state: GraphState) -> GraphState:
        user_text = _latest_user_text(state["messages"])
        if state.get("last_tool") == "generation_subgraph":
            payload = state.get("tool_payload") or {}
            if isinstance(payload, dict) and payload.get("status") == "failed":
                failure_reason = payload.get("failure_reason") or "No suitable new framework found."
                state["answer"] = failure_reason
                state["critic_decision"] = "finalize"
                state["critic_reason"] = "Generation subgraph failed"
                return state
        if _has_generation_candidates(state.get("traces", [])) and not _is_dissatisfied(user_text):
            state["critic_decision"] = "finalize"
            state["critic_reason"] = "Candidates already available from generation subgraph"
            return state

        if state["loop_count"] >= state["max_loops"]:
            state["answer"] = state.get("answer") or "Loop limit reached without sufficient results."
            state["critic_decision"] = "finalize"
            state["critic_reason"] = "Loop limit reached"
            return state

        if state.get("plan_error"):
            missing = _summarize_missing_fields(state.get("missing_fields", []))
            if missing:
                state["answer"] = f"Please provide missing inputs: {missing}."
                state["critic_decision"] = "finalize"
                state["critic_reason"] = "Missing required inputs"
                return state
        if state.get("missing_inputs"):
            missing = ", ".join(state["missing_inputs"])
            state["answer"] = f"Please provide {missing} to run the diffusion prediction."
            state["critic_decision"] = "finalize"
            state["critic_reason"] = "Missing diffusion inputs"
            return state

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You are the Final-level Critic. Decide whether to finalize or replan. "
                    "Return JSON with keys: decision (finalize|replan) and reason. "
                    "Reason MUST be a non-empty string.",
                ),
                (
                    "human",
                    "User request: {request}\nMessages: {messages}\nPlan: {plan}\nTraces: {traces}\nLoop: {loop}/{max_loops}\nReturn JSON only.",
                ),
            ]
        )
        chain = prompt | llm | StrOutputParser()
        raw = chain.invoke(
            {
                "request": user_text,
                "messages": json.dumps(_serialize_messages(state["messages"]), ensure_ascii=False),
                "plan": json.dumps(state.get("plan") or {}, ensure_ascii=False),
                "traces": json.dumps([t.as_dict() for t in state["traces"]], ensure_ascii=False),
                "loop": state["loop_count"],
                "max_loops": state["max_loops"],
            }
        )
        try:
            parsed = json.loads(_extract_json_block(raw))
            decision = parsed.get("decision", "finalize")
            reason = parsed.get("reason")
        except json.JSONDecodeError:
            decision = "finalize"
            reason = None

        state["critic_decision"] = decision
        state["critic_reason"] = reason
        if decision == "replan":
            state["loop_count"] += 1
        return state

    def finalize(state: GraphState) -> GraphState:
        if state.get("answer"):
            state["final_answer"] = state["answer"]
            return state

        draft = ""
        prompt = _finalize_prompt()
        chain = prompt | llm | StrOutputParser()
        final = chain.invoke(
            {
                "messages": json.dumps(_serialize_messages(state["messages"]), ensure_ascii=False),
                "memory": json.dumps(state["memory"], ensure_ascii=False),
                "plan": json.dumps(state.get("plan") or {}, ensure_ascii=False),
                "traces": json.dumps([t.as_dict() for t in state["traces"]], ensure_ascii=False),
                "draft": draft,
            }
        )
        final_text = (final or "").strip()
        state["final_answer"] = final_text or "No response generated."
        return state

    graph.add_node("classify", classify_intent)
    graph.add_node("plan", plan)
    graph.add_node("execute", execute)
    graph.add_node("critic", critic)
    graph.add_node("finalize", finalize)

    graph.set_entry_point("classify")
    graph.add_edge("classify", "plan")
    graph.add_edge("plan", "execute")
    graph.add_edge("execute", "critic")

    def decide_next(state: GraphState) -> str:
        if state.get("critic_decision") == "replan" and state["loop_count"] < state["max_loops"]:
            return "plan"
        return "finalize"

    graph.add_conditional_edges("critic", decide_next, {"plan": "plan", "finalize": "finalize"})
    graph.add_edge("finalize", END)
    return graph.compile()


def _to_message_dict(msg: BaseMessage) -> Dict[str, str]:
    role = "assistant" if isinstance(msg, AIMessage) else "user" if isinstance(msg, HumanMessage) else "system"
    return {"role": role, "content": msg.content}


def _to_chat_messages(messages: List[Dict[str, str]]) -> List[BaseMessage]:
    chat: List[BaseMessage] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "assistant":
            chat.append(AIMessage(content=content))
        elif role == "system":
            chat.append(SystemMessage(content=content))
        else:
            chat.append(HumanMessage(content=content))
    return chat


def run_zeoagent_conversation(
    messages: List[Dict[str, str]],
    state: Optional[AgentState] = None,
    search_backend: Optional[Any] = None,
    llm: Optional[BaseLanguageModel] = None,
    trace_writer: TraceWriter = None,
) -> Dict[str, Any]:
    """
    Execute a conversational turn through the replanning agent.
    """
    state = state or AgentState()
    llm = llm or build_general_llm()
    graph = build_agent_graph(llm, search_backend, trace_writer=trace_writer)

    chat_messages = _to_chat_messages(state.messages + messages)
    graph_state: GraphState = {
        "messages": chat_messages,
        "memory": dict(state.memory),
        "traces": list(state.traces),
        "intent": None,
        "intent_reason": None,
        "plan": None,
        "plan_obj": None,
        "plan_raw": None,
        "plan_error": None,
        "missing_fields": [],
        "missing_inputs": [],
        "loop_count": 0,
        "max_loops": 2,
        "critic_decision": None,
        "critic_reason": None,
        "answer": None,
        "final_answer": None,
        "last_tool": None,
        "tool_payload": {},
        "search_results": [],
        "reasoning": [],
    }
    result = graph.invoke(graph_state)

    final_answer = result.get("final_answer") or result.get("answer") or "No response generated."
    updated_messages = chat_messages + [AIMessage(content=final_answer)]
    state.messages = [_to_message_dict(m) for m in updated_messages]
    state.memory = result.get("memory", state.memory)
    state.traces = result.get("traces", state.traces)

    tool_name = result.get("last_tool") or result.get("intent") or result.get("memory", {}).get("last_tool")
    return {
        "answer": final_answer,
        "tool": tool_name,
        "result": result.get("tool_payload", {}),
        "plan": result.get("plan"),
        "critic": {
            "decision": result.get("critic_decision"),
            "reason": result.get("critic_reason"),
        },
        "results": {
            "papers": [p.model_dump() for p in result.get("search_results", [])],
            "recommendations": [],
        },
        "reasoning": result.get("reasoning", []),
        "traces": [t.as_dict() for t in state.traces],
        "state": state.snapshot(),
    }


__all__ = [
    "run_zeoagent_conversation",
    "preview_generation_plan",
    "AgentState",
    "SYSTEM_PROMPT",
    "build_qwen_llm",
    "build_generation_llm",
]
