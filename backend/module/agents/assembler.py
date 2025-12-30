from __future__ import annotations

import shutil
from pathlib import Path
from typing import Dict, Iterable, List

from backend.module.agents.agent_types import AgentsConfig, MCPServerConfig, ResolvedAgentConfig
from backend.module.config_handler import settings
from backend.module.logging import LoggingInterceptor

AGENTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = AGENTS_DIR.parents[2]
SNAPSHOT_ROOT = REPO_ROOT / "agents_snapshots"
DEFAULT_CONFIG_PATH = AGENTS_DIR / "agents_config.toml"
logger = LoggingInterceptor("agents_assembler")


def _quote(value: str) -> str:
    return f"\"{value.replace('\"', '\\\"')}\""


def _format_inline_table(data: Dict[str, str]) -> str:
    items = []
    for key in sorted(data):
        items.append(f"{key} = {_quote(str(data[key]))}")
    return "{ " + ", ".join(items) + " }"


def _format_list(values: Iterable[str]) -> str:
    return "[" + ", ".join(_quote(str(v)) for v in values) + "]"


def _render_model_provider_block(resolved: ResolvedAgentConfig) -> List[str]:
    provider = resolved.model_provider.lower()
    lines: List[str] = []

    if provider == "azure":
        lines.append("[model_providers.azure]")
        lines.append('name = "Azure"')
        base_url = str(settings.azure.api_base)
        if not base_url.rstrip("/").endswith("openai"):
            base_url = base_url.rstrip("/") + "/openai"
        lines.append(f"base_url = {_quote(base_url)}")
        lines.append('env_key = "AZURE_OPENAI_LLM_API_KEY"')
        lines.append(f'query_params = {{ api-version = {_quote(settings.azure.llm.api_version)} }}')
        lines.append('wire_api = "responses"')
    elif provider == "openai":
        lines.append("[model_providers.openai]")
        lines.append('name = "OpenAI"')
        lines.append(f'base_url = {_quote(str(settings.openai.api_base))}')
        lines.append('env_key = "OPENAI_API_KEY"')
        lines.append('wire_api = "responses"')
    else:
        lines.append(f"[model_providers.{provider}]")
        lines.append(f"name = {_quote(provider)}")
        lines.append('wire_api = "responses"')

    if resolved.model_reasoning_effort:
        lines.append(f"model_reasoning_effort = {_quote(resolved.model_reasoning_effort)}")
    if resolved.model_reasoning_summary:
        lines.append(f"model_reasoning_summary = {_quote(resolved.model_reasoning_summary)}")
    if resolved.request_max_retries is not None:
        lines.append(f"request_max_retries = {resolved.request_max_retries}")
    if resolved.stream_max_retries is not None:
        lines.append(f"stream_max_retries = {resolved.stream_max_retries}")

    return lines


def _render_mcp_server_block(name: str, server: MCPServerConfig) -> List[str]:
    lines = [f"[mcp_servers.{name}]"]
    if server.command:
        lines.append(f"command = {_quote(server.command)}")
    if server.cwd:
        lines.append(f"cwd = {_quote(server.cwd)}")
    if server.args:
        lines.append(f"args = {_format_list(server.args)}")
    if server.env:
        lines.append(f"env = {_format_inline_table(server.env)}")
    if server.env_vars:
        lines.append(f'env_vars = {_format_list(server.env_vars)}')
    if server.url:
        lines.append(f"url = {_quote(server.url)}")
    if server.bearer_token_env_var:
        lines.append(f"bearer_token_env_var = {_quote(server.bearer_token_env_var)}")
    if server.http_headers:
        lines.append(f"http_headers = {_format_inline_table(server.http_headers)}")
    if server.env_http_headers:
        lines.append(f"env_http_headers = {_format_inline_table(server.env_http_headers)}")
    if server.startup_timeout_sec is not None:
        lines.append(f"startup_timeout_sec = {server.startup_timeout_sec}")
    if server.tool_timeout_sec is not None:
        lines.append(f"tool_timeout_sec = {server.tool_timeout_sec}")
    if server.enabled is not None:
        lines.append(f"enabled = {'true' if server.enabled else 'false'}")
    if server.enabled_tools:
        lines.append(f"enabled_tools = {_format_list(server.enabled_tools)}")
    if server.disabled_tools:
        lines.append(f"disabled_tools = {_format_list(server.disabled_tools)}")
    return lines


def _render_agent_config(resolved: ResolvedAgentConfig) -> str:
    lines: List[str] = []
    lines.append(f"model = {_quote(resolved.model)}")
    lines.append(f"llm_profile = {_quote(resolved.llm_profile)}")
    lines.append(f"model_provider = {_quote(resolved.model_provider)}")
    if resolved.preferred_auth_method:
        lines.append(f"preferred_auth_method = {_quote(resolved.preferred_auth_method)}")
    if resolved.model_reasoning_effort:
        lines.append(f"model_reasoning_effort = {_quote(resolved.model_reasoning_effort)}")
    if resolved.model_reasoning_summary:
        lines.append(f"model_reasoning_summary = {_quote(resolved.model_reasoning_summary)}")
    if resolved.request_max_retries is not None:
        lines.append(f"request_max_retries = {resolved.request_max_retries}")
    if resolved.stream_max_retries is not None:
        lines.append(f"stream_max_retries = {resolved.stream_max_retries}")
    lines.append("")  # spacer
    lines.extend(_render_model_provider_block(resolved))

    if resolved.mcp_servers:
        lines.append("")
        for name, server in resolved.mcp_servers.items():
            lines.extend(_render_mcp_server_block(name, server))
            lines.append("")
        if lines and not lines[-1]:
            lines.pop()  # clean trailing spacer

    return "\n".join(lines) + "\n"


def _copy_instructions(resolved: ResolvedAgentConfig) -> Path:
    if not resolved.instructions_path.exists():
        raise FileNotFoundError(f"Instructions file not found: {resolved.instructions_path}")
    dest = resolved.agent_folder / "AGENTS.md"
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(resolved.instructions_path, dest)
    logger.info("Copied instructions", agent=resolved.name, destination=str(dest))
    return dest


def assemble_agents(config_path: Path | None = None) -> List[ResolvedAgentConfig]:
    """Load agents_config.toml and materialize per-agent folders and configs."""
    path = config_path or DEFAULT_CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(f"agents_config.toml not found at {path}")

    config = AgentsConfig.load(path)
    instructions_root = (AGENTS_DIR / config.defaults.instructions_dir).resolve()
    snapshot_root = SNAPSHOT_ROOT.resolve()
    snapshot_root.mkdir(parents=True, exist_ok=True)
    resolved_agents = [
        ResolvedAgentConfig.from_spec(
            snapshot_root=snapshot_root,
            instructions_root=instructions_root,
            defaults=config.defaults,
            spec=spec,
        )
        for spec in config.agents
    ]

    for agent in resolved_agents:
        agent.agent_folder.mkdir(parents=True, exist_ok=True)
        _copy_instructions(agent)
        config_body = _render_agent_config(agent)
        config_path = agent.agent_folder / "config.toml"
        config_path.write_text(config_body, encoding="utf-8")
        logger.info(
            "Assembled agent",
            agent=agent.name,
            folder=str(agent.agent_folder),
            config=str(config_path),
            instructions=str(agent.instructions_path),
        )

    return resolved_agents


__all__ = ["assemble_agents"]
