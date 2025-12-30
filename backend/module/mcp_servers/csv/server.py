from __future__ import annotations

import sys
from typing import Any, Callable, Dict, List, Optional

from pathlib import Path

try:
    from fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit("fastmcp is required to run the CSV MCP server") from exc

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.module.mcp_servers.csv import tools


mcp = FastMCP(
    name="csv-mcp",
    instructions="CSV exploration tools for parquet-backed uploads.",
)


@mcp.tool()
def list_parquet_ids(user_id: str) -> List[str]:
    """Return parquet file_ids for a user."""
    return tools.list_user_parquet_ids(user_id)


@mcp.tool()
def sample_data(file_id: str, position: str = "head", limit: int = 5) -> List[Dict[str, Any]]:
    """Sample rows from a parquet dataset."""
    return tools.data_sample(file_id, position=position, limit=limit)


@mcp.tool()
def column_stats(file_id: str, column_name: str) -> Dict[str, Any]:
    """Get column statistics for a parquet dataset."""
    return tools.get_columns_stats(file_id, column_name)


@mcp.tool()
def verify_parquet_function(file_id: str, func_source: str) -> Dict[str, Any]:
    """
    Execute a user-provided function against the dataframe.

    func_source should be Python code defining a function named `func(df)`.
    """
    namespace: Dict[str, Any] = {}
    exec(func_source, namespace)  # noqa: S102
    func: Callable[[Any], Any] = namespace.get("func")
    return tools.verify_function(file_id, func)


if __name__ == "__main__":
    mcp.run()
