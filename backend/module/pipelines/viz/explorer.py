from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.module.logging import LoggingInterceptor

logger = LoggingInterceptor("pipelines_viz_explorer")

DEFAULT_INDEX_PATH = Path(__file__).resolve().parents[3] / "test_files" / "echarts_schema.index.json"


@dataclass
class EChartsSchema:
    """Container for indexed ECharts schema data."""

    series_list: List[str]
    component_list: List[str]
    series_types: Dict[str, Any]
    components: Dict[str, Any]
    search_index: Dict[str, List[Tuple[str, str]]]


def _load_index_sync(index_path: Path) -> EChartsSchema:
    if not index_path.exists():
        raise FileNotFoundError(f"Schema index not found at {index_path}")
    with index_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return EChartsSchema(
        series_list=payload.get("series_list", []),
        component_list=payload.get("component_list", []),
        series_types=payload.get("series_types", {}),
        components=payload.get("components", {}),
        search_index=payload.get("search_index", {}),
    )


@lru_cache(maxsize=1)
def get_viz_index(index_path: Path = DEFAULT_INDEX_PATH) -> EChartsSchema:
    """Load and cache the viz schema index."""
    schema = _load_index_sync(index_path)
    logger.info("Viz schema loaded", series=len(schema.series_list), components=len(schema.component_list))
    return schema


async def get_viz_index_async(index_path: Path = DEFAULT_INDEX_PATH) -> EChartsSchema:
    """Async-friendly loader for the viz schema index."""
    return await asyncio.to_thread(get_viz_index, index_path)


def list_series_types(index_path: Path = DEFAULT_INDEX_PATH) -> List[str]:
    return get_viz_index(index_path).series_list


def list_components(index_path: Path = DEFAULT_INDEX_PATH) -> List[str]:
    return get_viz_index(index_path).component_list


def search_schema(keyword: str, limit: int = 20, index_path: Path = DEFAULT_INDEX_PATH) -> List[Dict[str, str]]:
    """Search components/properties by keyword."""
    schema = get_viz_index(index_path)
    key = keyword.lower().strip()
    if not key:
        return []
    results: List[Tuple[str, str]] = []
    if key in schema.search_index:
        results.extend(schema.search_index[key])
    for term, vals in schema.search_index.items():
        if key in term and term != key:
            results.extend(vals)
        if len(results) >= limit:
            break
    deduped: List[Dict[str, str]] = []
    seen = set()
    for kind, path in results[:limit]:
        if (kind, path) in seen:
            continue
        seen.add((kind, path))
        deduped.append({"type": kind, "path": path})
    logger.info("Schema search", keyword=keyword, results=len(deduped))
    return deduped


def summarize_series(series_type: str, index_path: Path = DEFAULT_INDEX_PATH) -> Dict[str, Any]:
    schema = get_viz_index(index_path)
    entry = schema.series_types.get(series_type)
    if not entry:
        raise ValueError(f"Series type '{series_type}' not found")
    props = entry.get("properties", {}) or {}
    summary_props = {}
    priority = [
        "data",
        "type",
        "name",
        "itemStyle",
        "label",
        "emphasis",
        "smooth",
        "stack",
        "areaStyle",
        "symbol",
        "symbolSize",
    ]
    for key in priority:
        if key in props:
            summary_props[key] = props[key]
    for key in list(props.keys()):
        if len(summary_props) >= 10:
            break
        if key not in summary_props:
            summary_props[key] = props[key]
    return {
        "series_type": series_type,
        "description": entry.get("description", ""),
        "property_count": len(props),
        "key_properties": summary_props,
    }


def validate_option(option: Dict[str, Any], index_path: Path = DEFAULT_INDEX_PATH) -> Dict[str, Any]:
    """
    Lightweight validation: checks series types exist and flags unknown properties.
    """
    schema = get_viz_index(index_path)
    warnings: List[str] = []
    errors: List[str] = []
    series_entries = option.get("series") or []
    if isinstance(series_entries, dict):
        series_entries = [series_entries]
    for idx, series_cfg in enumerate(series_entries):
        if not isinstance(series_cfg, dict):
            errors.append(f"series[{idx}] must be an object")
            continue
        series_type = series_cfg.get("type")
        if not series_type:
            errors.append(f"series[{idx}] missing 'type'")
            continue
        spec = schema.series_types.get(series_type)
        if not spec:
            errors.append(f"Unknown series type: {series_type}")
            continue
        known_props = spec.get("properties", {}) or {}
        for key in series_cfg.keys():
            if key not in known_props and key != "type":
                warnings.append(f"series[{idx}] unknown property '{key}'")
    return {"valid": not errors, "errors": errors, "warnings": warnings}

