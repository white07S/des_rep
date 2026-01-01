from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Ensure project root is importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel, Field

from backend.module.codex_agents.framework.utils import codex_safe_schema, parse_structured_response, run_agent_once
from backend.module.logging import LoggingInterceptor

logger = LoggingInterceptor("agents_db_agent_demo")

ResponseType = Literal["nl-sql", "nl-explore"]


class AgentResponse(BaseModel):
    """Structured response emitted by the database agent."""

    misleading: bool = Field(description="True if request is misleading or cannot be answered")
    response_type: Optional[ResponseType] = Field(
        default=None,
        description="Type of response: 'nl-sql' or 'nl-explore'. None if misleading=True",
    )
    reasoning: List[str] = Field(description="Step-by-step reasoning for the response")
    sql: str = Field(default="", description="SQL query without markdown fences for nl-sql responses")
    text: str = Field(default="", description="Exploratory text for nl-explore responses")


def get_agent_response_json_schema() -> Dict[str, Any]:
    """Generate Codex-safe JSON Schema for AgentResponse."""
    return codex_safe_schema(AgentResponse)


def parse_agent_response(text: str) -> AgentResponse:
    """Parse the final agent message as AgentResponse, tolerating extra text."""
    return parse_structured_response(text, AgentResponse)


async def run_demo() -> None:
    """Run a demo prompt against the assembled db_agent."""
    prompt = "we are going to be working with demo_user, tell me : User query: what is bank of america total assets?"

    output_schema = get_agent_response_json_schema()
    result = await run_agent_once("db_agent", prompt, output_schema=output_schema, on_event=lambda e: print(json.dumps(e, ensure_ascii=False)))

    if result.final_message:
        try:
            parsed = parse_agent_response(result.final_message)
            logger.info(
                "Parsed agent response",
                misleading=parsed.misleading,
                response_type=parsed.response_type,
                reasoning=" | ".join(parsed.reasoning),
            )
        except Exception as exc:
            logger.error("Failed to parse agent response", error=str(exc))


if __name__ == "__main__":
    asyncio.run(run_demo())
