from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Dict, Optional, Type, TypeVar

from pydantic import BaseModel

from backend.module.codex_agents.framework.assembler import assemble_agents
from backend.module.codex_agents.framework.agent_types import ResolvedAgentConfig
from backend.module.codex_agents.framework.sessions import create_new_session
from backend.module.logging import LoggingInterceptor

logger = LoggingInterceptor("agents_utils")

T = TypeVar("T", bound=BaseModel)


def parse_structured_response(text: str, model_cls: Type[T]) -> T:
    """
    Parse a structured model from agent text output, tolerating extra text.
    Attempts direct JSON, then brace extraction, then regex fallback.
    """
    cleaned = text.strip()
    try:
        data = json.loads(cleaned)
        return model_cls(**data)
    except json.JSONDecodeError:
        pass

    start_idx = cleaned.find("{")
    if start_idx != -1:
        brace_count = 0
        end_idx = start_idx
        for i, char in enumerate(cleaned[start_idx:], start_idx):
            if char == "{":
                brace_count += 1
            elif char == "}":
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        if end_idx > start_idx:
            try:
                data = json.loads(cleaned[start_idx:end_idx])
                return model_cls(**data)
            except Exception:
                pass

    match = re.search(r"\{[^{}]*\"misleading\"[^{}]*\}", cleaned, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            return model_cls(**data)
        except Exception as exc:
            raise ValueError(f"Could not parse structured response via regex: {exc}") from exc
    raise ValueError(f"Could not parse structured response from text: {cleaned[:500]}...")


def codex_safe_schema(model_cls: Type[BaseModel]) -> Dict[str, Any]:
    """
    Build a Codex-safe JSON schema from a Pydantic model, ensuring
    additionalProperties defaults to false and required includes all properties.
    """
    raw = model_cls.model_json_schema()

    def _fix(node: Any) -> Any:
        if isinstance(node, dict):
            if node.get("type") == "object":
                props = node.setdefault("properties", {})
                node["required"] = list(props.keys())
                node["additionalProperties"] = False
            if node.get("type") == "array" and isinstance(node.get("items"), dict):
                _fix(node["items"])
            for key, val in list(node.items()):
                if isinstance(val, (dict, list)):
                    _fix(val)
        elif isinstance(node, list):
            for item in node:
                _fix(item)
        return node

    return _fix(raw)


@dataclass
class AgentRunResult:
    final_message: Optional[str]
    events: list[Dict[str, Any]]


async def run_agent_once(
    agent_name: str,
    prompt: str,
    output_schema: Optional[Dict[str, Any]] = None,
    on_event: Optional[Callable[[Dict[str, Any]], None]] = None,
    config_path: Optional[Path] = None,
) -> AgentRunResult:
    """
    Stream a single agent run and return the final agent_message text.
    Does not coalesce events; callers can stream via on_event for atomic handling.
    """
    events: list[Dict[str, Any]] = []
    final_message: Optional[str] = None
    async for event in create_new_session(agent_name, prompt, output_schema=output_schema, config_path=config_path):
        events.append(event)
        if on_event:
            on_event(event)
        if event.get("type") == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                final_message = item.get("text")
    return AgentRunResult(final_message=final_message, events=events)


def get_agent_config(agent_name: str) -> ResolvedAgentConfig:
    """Find a resolved agent config by name."""
    for agent in assemble_agents():
        if agent.name == agent_name:
            return agent
    raise ValueError(f"Agent '{agent_name}' not found in agents_config.toml")
