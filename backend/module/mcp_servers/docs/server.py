from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover
    raise SystemExit("fastmcp is required to run the Docs MCP server") from exc

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.module.mcp_servers.docs import tools

mcp = FastMCP(
    name="docs-mcp",
    instructions="Document search tools over processed PDF artifacts.",
)


@mcp.tool()
def list_documents(user_id: Optional[str] = None) -> List[Dict[str, Any]]:
    """List processed PDF documents (file_id, user_id, original filename)."""
    return tools.list_docs(user_id)


@mcp.tool()
async def search_document(
    query: str,
    file_id: str,
    user_id: Optional[str] = None,
    provider: Optional[str] = None,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """Hybrid search over a document, returning top contexts with neighbor chunks."""
    return await tools.search_docs(
        query,
        file_id=file_id,
        user_id=user_id,
        provider=provider,
        top_k=top_k,
    )


if __name__ == "__main__":
    mcp.run()
