"""Local parsing and retrieval helpers for separation-oriented tasks."""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


_SPECIES_ALIASES: Dict[str, Tuple[str, ...]] = {
    "H2": ("h2", "hydrogen"),
    "CO2": ("co2", "carbon dioxide"),
    "CH4": ("ch4", "methane"),
    "N2": ("n2", "nitrogen"),
    "O2": ("o2", "oxygen"),
    "CO": ("co", "carbon monoxide"),
    "HE": ("he", "helium"),
}

_STOPWORDS = {
    "a",
    "an",
    "and",
    "for",
    "the",
    "to",
    "of",
    "in",
    "on",
    "with",
    "that",
    "is",
    "are",
    "be",
    "from",
    "or",
    "as",
    "by",
    "it",
    "this",
    "at",
}


def is_separation_request(text: str) -> bool:
    low = (text or "").lower()
    keywords = (
        "separation",
        "separate",
        "gas pair",
        "permselectivity",
        "co2/ch4",
        "co2 ch4",
        "co2 and ch4",
    )
    return any(token in low for token in keywords)


def canonicalize_species(raw: str) -> Optional[str]:
    value = (raw or "").strip().lower().replace("-", " ")
    value = re.sub(r"\s+", " ", value)
    if not value:
        return None
    for species, aliases in _SPECIES_ALIASES.items():
        if value == species.lower() or value in aliases:
            return species
    return None


def extract_species_mentions(text: str) -> List[str]:
    low = (text or "").lower()
    positions: List[Tuple[int, str]] = []
    for species, aliases in _SPECIES_ALIASES.items():
        for alias in aliases:
            pattern = rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])"
            for match in re.finditer(pattern, low):
                positions.append((match.start(), species))
    positions.sort(key=lambda item: item[0])
    ordered: List[str] = []
    seen: set[str] = set()
    for _, species in positions:
        if species in seen:
            continue
        seen.add(species)
        ordered.append(species)
    return ordered


@lru_cache(maxsize=8)
def _load_diameter_table_cached(path_str: str) -> Dict[str, Dict[str, Any]]:
    path = Path(path_str)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data.get("species", data if isinstance(data, list) else [])
    table: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        species = canonicalize_species(str(row.get("species") or ""))
        if not species:
            continue
        value = row.get("kinetic_diameter_A")
        try:
            diameter = float(value)
        except (TypeError, ValueError):
            continue
        table[species] = {
            "species": species,
            "kinetic_diameter_A": diameter,
            "source": row.get("source", ""),
            "aliases": row.get("aliases", []),
        }
    return table


def load_molecular_diameter_table(path: Path) -> Dict[str, Dict[str, Any]]:
    return _load_diameter_table_cached(str(path.resolve()))


def resolve_species_diameter(
    species: Optional[str], table: Dict[str, Dict[str, Any]]
) -> Optional[Dict[str, Any]]:
    canonical = canonicalize_species(species or "")
    if not canonical:
        return None
    return table.get(canonical)


def infer_permeate_species(
    user_request: str,
    table: Dict[str, Dict[str, Any]],
) -> Optional[str]:
    low = (user_request or "").lower()
    mentions = extract_species_mentions(user_request)
    if not mentions:
        return None

    explicit_patterns = [
        r"(?:permeate|pass|diffuse|transport)\s+(?:of\s+)?([a-z0-9\s\-]+)",
        r"(?:remove|capture)\s+([a-z0-9\s\-]+)",
        r"([a-z0-9\s\-]+)\s+(?:should|must)\s+(?:pass|permeate|diffuse)",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, low)
        if not match:
            continue
        candidate = canonicalize_species(match.group(1))
        if candidate and candidate in table:
            return candidate

    ranked: List[Tuple[float, str]] = []
    for species in mentions:
        entry = table.get(species)
        if not entry:
            continue
        ranked.append((float(entry["kinetic_diameter_A"]), species))
    if ranked:
        ranked.sort(key=lambda item: item[0])
        return ranked[0][1]
    return mentions[0]


def infer_retentate_species(
    user_request: str,
    table: Dict[str, Dict[str, Any]],
    permeate_species: Optional[str] = None,
) -> Optional[str]:
    low = (user_request or "").lower()
    mentions = extract_species_mentions(user_request)
    if not mentions:
        return None

    explicit_patterns = [
        r"(?:block|reject|exclude|hinder|suppress|prevent|retain)\s+(?:of\s+)?([a-z0-9\s\-]+)",
        r"([a-z0-9\s\-]+)\s+(?:should|must)\s+(?:be\s+)?(?:blocked|rejected|excluded|retained)",
    ]
    for pattern in explicit_patterns:
        match = re.search(pattern, low)
        if not match:
            continue
        candidate = canonicalize_species(match.group(1))
        if candidate and candidate in table:
            return candidate

    permeate = canonicalize_species(permeate_species or "")
    ranked: List[Tuple[float, str]] = []
    for species in mentions:
        if permeate and species == permeate:
            continue
        entry = table.get(species)
        if not entry:
            continue
        ranked.append((float(entry["kinetic_diameter_A"]), species))
    if ranked:
        ranked.sort(key=lambda item: item[0], reverse=True)
        return ranked[0][1]

    if permeate:
        for species in mentions:
            if species != permeate:
                return species
        return None

    ranked_all: List[Tuple[float, str]] = []
    for species in mentions:
        entry = table.get(species)
        if not entry:
            continue
        ranked_all.append((float(entry["kinetic_diameter_A"]), species))
    if ranked_all:
        ranked_all.sort(key=lambda item: item[0], reverse=True)
        return ranked_all[0][1]
    return mentions[-1]


def infer_gas_pair(user_request: str) -> Optional[str]:
    mentions = extract_species_mentions(user_request)
    if len(mentions) < 2:
        return None
    return f"{mentions[0]}/{mentions[1]}"


def _tokenize(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9]+", (text or "").lower()) if token]


def _find_corpus_files(corpus_dir: Path) -> List[Path]:
    if not corpus_dir.exists():
        return []
    primary = corpus_dir / "separation_corpus.jsonl"
    if primary.exists():
        return [primary]
    clean = sorted(corpus_dir.glob("*separation*.clean.jsonl"))
    if clean:
        return clean
    fallback = sorted(corpus_dir.glob("*separation*.jsonl"))
    return fallback


def _load_corpus_records(corpus_dir: Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for path in _find_corpus_files(corpus_dir):
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict) or not obj.get("text"):
                continue
            obj["_source_path"] = str(path)
            records.append(obj)
    return records


def hit_identifier(hit: Dict[str, Any]) -> str:
    chunk_id = str(hit.get("chunk_id") or "").strip()
    if chunk_id:
        return chunk_id
    source_file = str(hit.get("source_file") or hit.get("_source_path") or "").strip()
    page = hit.get("page")
    if source_file:
        return f"{source_file}::page={page}"
    text = re.sub(r"\s+", " ", str(hit.get("text", ""))).strip()
    if text:
        return f"text::{text[:120]}"
    return "unknown"


def retrieve_separation_evidence(
    query: str,
    corpus_dir: Path,
    top_k: int = 4,
    excluded_hit_ids: Optional[Iterable[str]] = None,
) -> List[Dict[str, Any]]:
    records = _load_corpus_records(corpus_dir)
    if not records:
        return []

    excluded = {str(item).strip() for item in (excluded_hit_ids or []) if str(item).strip()}
    query_terms = [token for token in _tokenize(query) if token not in _STOPWORDS]
    if not query_terms:
        return []

    tokenized_docs: List[List[str]] = [_tokenize(str(rec.get("text", ""))) for rec in records]
    doc_lengths = [len(doc) for doc in tokenized_docs]
    avgdl = (sum(doc_lengths) / len(doc_lengths)) if doc_lengths else 1.0
    if avgdl <= 0:
        avgdl = 1.0

    doc_freq: Counter[str] = Counter()
    for doc in tokenized_docs:
        for term in set(doc):
            doc_freq[term] += 1

    k1 = 1.5
    b = 0.75
    total_docs = len(records)
    scored: List[Tuple[float, Dict[str, Any]]] = []
    for rec, doc_tokens, doc_len in zip(records, tokenized_docs, doc_lengths):
        tf = Counter(doc_tokens)
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq <= 0:
                continue
            df = doc_freq.get(term, 0)
            idf = math.log(1.0 + ((total_docs - df + 0.5) / (df + 0.5)))
            denom = freq + k1 * (1.0 - b + b * (doc_len / avgdl))
            score += idf * ((freq * (k1 + 1.0)) / denom)
        if score > 0:
            result = dict(rec)
            if hit_identifier(result) in excluded:
                continue
            result["score"] = round(score, 6)
            scored.append((score, result))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in scored[: max(0, top_k)]]


def format_retrieval_context(hits: List[Dict[str, Any]], max_chars: int = 1400) -> str:
    if not hits:
        return ""
    lines: List[str] = []
    for hit in hits:
        chunk_id = hit.get("chunk_id", "unknown")
        page = hit.get("page", "?")
        frameworks = ", ".join(hit.get("frameworks", []) or [])
        gas_pairs = ", ".join(hit.get("gas_pairs", []) or [])
        text = re.sub(r"\s+", " ", str(hit.get("text", ""))).strip()
        snippet = text[:220]
        lines.append(
            f"[{chunk_id}] page={page}; frameworks={frameworks or 'n/a'}; gas_pairs={gas_pairs or 'n/a'}; snippet={snippet}"
        )
    context = "\n".join(lines)
    if len(context) > max_chars:
        return context[: max_chars - 3] + "..."
    return context


def compact_hits_for_trace(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    compact: List[Dict[str, Any]] = []
    for hit in hits:
        compact.append(
            {
                "chunk_id": hit.get("chunk_id"),
                "page": hit.get("page"),
                "frameworks": hit.get("frameworks", []),
                "gas_pairs": hit.get("gas_pairs", []),
                "score": hit.get("score"),
                "source_file": hit.get("source_file"),
            }
        )
    return compact
