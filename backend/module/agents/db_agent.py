from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Ensure project root is importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pydantic import BaseModel, Field

from backend.module.agents.assembler import assemble_agents
from backend.module.agents.sessions import create_new_session
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
    """Generate JSON Schema for AgentResponse."""
    return {
        "type": "object",
        "title": "AgentResponse",
        "description": "Agent response supporting misleading detection, SQL generation, and database exploration.",
        "properties": {
            "misleading": {"type": "boolean"},
            "response_type": {"type": ["string", "null"], "enum": ["nl-sql", "nl-explore", None]},
            "reasoning": {"type": "array", "items": {"type": "string"}},
            "sql": {"type": "string"},
            "text": {"type": "string"},
        },
        "required": ["misleading", "response_type", "reasoning", "sql", "text"],
        "additionalProperties": False,
    }


def parse_agent_response(text: str) -> AgentResponse:
    """Parse the final agent message as AgentResponse, tolerating extra text."""
    text = text.strip()
    try:
        data = json.loads(text)
        return AgentResponse(**data)
    except json.JSONDecodeError:
        pass

    start_idx = text.find("{")
    if start_idx != -1:
        brace_count = 0
        end_idx = start_idx
        for i, c in enumerate(text[start_idx:], start_idx):
            if c == "{":
                brace_count += 1
            elif c == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        if end_idx > start_idx:
            try:
                data = json.loads(text[start_idx:end_idx])
                return AgentResponse(**data)
            except Exception:
                pass
    raise ValueError(f"Could not parse AgentResponse from text: {text[:500]}...")


async def run_demo() -> None:
    """Run a demo prompt against the assembled db_agent."""
    assemble_agents()  # ensure configs/materialization are up to date
    prompt = "we are going to be working with demo_user, tell me : User query: what is bank of america total assets?"

    output_schema = get_agent_response_json_schema()
    agent_message_text: Optional[str] = None

    async for event in create_new_session(
        "db_agent",
        prompt,
        output_schema=output_schema,
    ):
        print(json.dumps(event, ensure_ascii=False))
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                agent_message_text = item.get("text")

    if agent_message_text:
        try:
            parsed = parse_agent_response(agent_message_text)
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
