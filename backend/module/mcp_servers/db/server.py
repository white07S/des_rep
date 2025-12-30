from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit("fastmcp is required to run the DB MCP server") from exc

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.module.mcp_servers.db import tools

mcp = FastMCP(
    name="db-mcp",
    instructions="SQLite schema exploration tools for uploaded databases.",
)


@mcp.tool()
def list_databases(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """List available SQLite artifacts (file_id + user_id)."""
    return tools.list_databases(user_id)


@mcp.tool()
def list_tables(file_id: str, user_id: Optional[str] = None) -> List[str]:
    """List tables in a SQLite artifact."""
    return tools.list_tables(file_id, user_id)


@mcp.tool()
def sample_table(
    file_id: str,
    table_name: str,
    position: str = "head",
    limit: int = 5,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Sample rows from a table."""
    return tools.sample_data_table(
        file_id=file_id,
        table_name=table_name,
        user_id=user_id,
        position=position,
        limit=limit,
    )


@mcp.tool()
def column_stats(
    file_id: str,
    table_name: str,
    column_name: str,
    user_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Column statistics for a table."""
    return tools.get_column_stats(file_id, table_name, column_name, user_id=user_id)


@mcp.tool()
def table_details(file_id: str, table_name: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Full table details: columns and foreign keys."""
    return tools.get_table_details(file_id, table_name, user_id=user_id)


@mcp.tool()
def relationship_graph(
    file_id: str,
    user_id: Optional[str] = None,
    include_columns: bool = False,
) -> Dict[str, Any]:
    """Relationship graph built from ai_spec_schema.json (or PRAGMA fallback)."""
    return tools.get_relationship_graph(file_id, user_id=user_id, include_columns=include_columns)


@mcp.tool()
def validate_query(file_id: str, sql: str, user_id: Optional[str] = None) -> Dict[str, Any]:
    """Validate read-only SQL query using EXPLAIN QUERY PLAN."""
    return tools.validate_query(file_id, sql, user_id=user_id)


if __name__ == "__main__":
    mcp.run()
