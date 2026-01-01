from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# Ensure project root is importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.module.codex_agents.framework.utils import codex_safe_schema, parse_structured_response, run_agent_once
from backend.module.logging import LoggingInterceptor

logger = LoggingInterceptor("agents_doc_agent")

AGENT_NAME = "doc_agent"


class DocAgentResponse(BaseModel):
    """Document agent response with misleading flag, reasoning, and final text."""

    misleading: bool = Field(description="True if request is misleading or cannot be answered")
    reasoning: List[str] = Field(description="Step-by-step reasoning for the response")
    text: str = Field(default="", description="Final answer or clarification text")


def get_doc_response_json_schema() -> Dict[str, Any]:
    """Generate Codex-safe JSON Schema for DocAgentResponse."""
    return codex_safe_schema(DocAgentResponse)


def parse_doc_response(text: str) -> DocAgentResponse:
    """Parse final agent message into DocAgentResponse."""
    return parse_structured_response(text, DocAgentResponse)


async def run_demo() -> None:
    """Run a demo prompt against the assembled doc_agent."""
    prompt = "User: demo_user querying (use mcp) : what is the net income and some other numbers for Bank of America? explain the results."
    schema = get_doc_response_json_schema()
    result = await run_agent_once(AGENT_NAME, prompt, output_schema=schema, on_event=lambda e: print(json.dumps(e, ensure_ascii=False)))

    if not result.final_message:
        logger.error("No agent message received from doc agent session")
        return

    try:
        parsed = parse_doc_response(result.final_message)
        logger.info("Parsed doc agent response", misleading=parsed.misleading, reasoning=" | ".join(parsed.reasoning))
    except Exception as exc:
        logger.error("Failed to parse doc agent response", error=str(exc))


if __name__ == "__main__":
    asyncio.run(run_demo())
