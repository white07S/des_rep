from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type

from pydantic import BaseModel, create_model

# Ensure project root is importable when run as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.module.codex_agents.framework.utils import codex_safe_schema, parse_structured_response, run_agent_once
from backend.module.logging import LoggingInterceptor

logger = LoggingInterceptor("agents_response_agent")

AGENT_NAME = "response_agent"


def build_dynamic_model(key1: Tuple[str, Type[Any]], key2: Tuple[str, Type[Any]]) -> Type[BaseModel]:
    """
    Build a Pydantic model with two required keys for dynamic responses.
    key tuples: (field_name, field_type)
    """
    fields = {
        key1[0]: (key1[1], ...),
        key2[0]: (key2[1], ...),
    }
    model = create_model("DynamicResponse", **fields)  # type: ignore[arg-type]
    return model


async def run_response_agent(
    prompt: str,
    key1: Tuple[str, Type[Any]],
    key2: Tuple[str, Type[Any]],
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
) -> BaseModel:
    """
    Run the response agent with a dynamic two-key schema.
    """
    model_cls = build_dynamic_model(key1, key2)
    schema = codex_safe_schema(model_cls)
    result = await run_agent_once(AGENT_NAME, prompt, output_schema=schema, on_event=on_event)
    if not result.final_message:
        raise RuntimeError("No agent message received from response agent session")
    return parse_structured_response(result.final_message, model_cls)


async def run_demo() -> None:
    """
    Demo: ask for a structured summary with custom keys.
    """
    prompt = "Provide a concise summary and a next_action."
    response = await run_response_agent(
        prompt,
        key1=("summary", str),
        key2=("next_action", str),
        on_event=lambda e: print(json.dumps(e, ensure_ascii=False)),
    )
    logger.info("Parsed response agent output", data=response.model_dump())


if __name__ == "__main__":
    asyncio.run(run_demo())
