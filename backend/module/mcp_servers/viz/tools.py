from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.module.logging import LoggingInterceptor
from backend.module.pipelines.viz import (
    get_viz_index,
    list_components,
    list_series_types,
    search_schema as search_schema_index,
    summarize_series,
    validate_option,
)

logger = LoggingInterceptor("mcp_viz_tools")


def ensure_index_available() -> Path:
    index_path = Path(__file__).resolve().parents[4] / "backend" / "module" / "viz_context" / "echarts_schema.index.json"
    if not index_path.exists():
        raise FileNotFoundError(
            f"ECharts schema index not found at {index_path}. Copy schema.index.json into backend/module/viz_context."
        )
    return index_path


def list_chart_types() -> List[str]:
    """List available ECharts series types from the indexed schema."""
    path = ensure_index_available()
    series = list_series_types(path)
    logger.info("Listed chart types", count=len(series))
    return series


def list_chart_components() -> List[str]:
    """List available ECharts components from the indexed schema."""
    path = ensure_index_available()
    comps = list_components(path)
    logger.info("Listed chart components", count=len(comps))
    return comps


def search_schema(keyword: str, limit: int = 10) -> List[Dict[str, str]]:
    """Search series/components/properties by keyword."""
    path = ensure_index_available()
    results = search_schema_index(keyword, limit=limit, index_path=path)
    logger.info("Schema search completed", keyword=keyword, results=len(results))
    return results


def get_series_summary(series_type: str) -> Dict[str, Any]:
    """Return a concise summary for a series type."""
    path = ensure_index_available()
    summary = summarize_series(series_type, index_path=path)
    logger.info("Series summary generated", series_type=series_type)
    return summary


def validate_chart_option(option: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight validation for an ECharts option dict."""
    path = ensure_index_available()
    result = validate_option(option, index_path=path)
    logger.info("Validated chart option", valid=result["valid"], errors=len(result["errors"]), warnings=len(result["warnings"]))
    return result


def _run_smoke_tests() -> None:
    try:
        path = ensure_index_available()
        print(f"[test] using index at {path}")
    except Exception as exc:  # noqa: BLE001
        print(f"[test] index missing: {exc}")
        return

    try:
        series = list_chart_types()
        print(f"[test] list_chart_types -> {len(series)} types; first={series[0] if series else 'none'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[test] list_chart_types error: {exc}")
        return

    try:
        comps = list_chart_components()
        print(f"[test] list_chart_components -> {len(comps)} components; first={comps[0] if comps else 'none'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[test] list_chart_components error: {exc}")
        return

    try:
        search_results = search_schema("line", limit=5)
        print(f"[test] search_schema('line') -> {search_results[:3]}")
    except Exception as exc:  # noqa: BLE001
        print(f"[test] search_schema error: {exc}")

    try:
        if series:
            summary = get_series_summary(series[0])
            print(f"[test] get_series_summary -> keys={list(summary.keys())}")
    except Exception as exc:  # noqa: BLE001
        print(f"[test] get_series_summary error: {exc}")

    try:
        sample_option = {"series": [{"type": series[0] if series else "line", "data": [1, 2, 3]}]}
        validation = validate_chart_option(sample_option)
        print(f"[test] validate_chart_option -> valid={validation['valid']} warnings={validation['warnings']}")
    except Exception as exc:  # noqa: BLE001
        print(f"[test] validate_chart_option error: {exc}")


if __name__ == "__main__":
    _run_smoke_tests()
