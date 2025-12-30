from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List

try:
    from fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit("fastmcp is required to run the Viz MCP server") from exc

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.module.mcp_servers.viz import tools

mcp = FastMCP(
    name="viz-mcp",
    instructions="ECharts schema exploration tools.",
)


@mcp.tool()
def list_chart_types() -> List[str]:
    """List available ECharts series types."""
    return tools.list_chart_types()


@mcp.tool()
def list_chart_components() -> List[str]:
    """List available ECharts components."""
    return tools.list_chart_components()


@mcp.tool()
def search_schema(keyword: str, limit: int = 10) -> List[Dict[str, str]]:
    """Search schema for components/properties by keyword."""
    return tools.search_schema(keyword, limit=limit)


@mcp.tool()
def series_summary(series_type: str) -> Dict[str, Any]:
    """Summarize a series type from the schema index."""
    return tools.get_series_summary(series_type)


@mcp.tool()
def validate_option(option: Dict[str, Any]) -> Dict[str, Any]:
    """Lightweight validation of an ECharts option."""
    return tools.validate_chart_option(option)


if __name__ == "__main__":
    mcp.run()
