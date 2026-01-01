from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import tomllib
from pydantic import BaseModel, ConfigDict, Field, model_validator

ENV_PATTERN = re.compile(r"\$\{env:([A-Z0-9_]+)(?:\|([^}]+))?}")


def _expand_env(value: Any) -> Any:
    """Recursively expand ${env:VAR|default} placeholders."""
    if isinstance(value, str):
        def repl(match: re.Match[str]) -> str:
            var, default = match.group(1), match.group(2) or ""
            return os.getenv(var, default)
        return ENV_PATTERN.sub(repl, value)
    if isinstance(value, dict):
        return {k: _expand_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_env(v) for v in value]
    return value


class MCPServerConfig(BaseModel):
    """Configuration for a single MCP server (stdio or HTTP)."""

    model_config = ConfigDict(extra="forbid")

    # stdio transport
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    cwd: Optional[str] = None
    env: Dict[str, str] = Field(default_factory=dict)
    env_vars: List[str] = Field(default_factory=list)

    # HTTP transport
    url: Optional[str] = None
    bearer_token_env_var: Optional[str] = None
    http_headers: Dict[str, str] = Field(default_factory=dict)
    env_http_headers: Dict[str, str] = Field(default_factory=dict)

    # common options
    startup_timeout_sec: Optional[int] = None
    tool_timeout_sec: Optional[int] = None
    enabled: Optional[bool] = None
    enabled_tools: List[str] = Field(default_factory=list)
    disabled_tools: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_transport(self) -> "MCPServerConfig":
        # Allow empty to support agents without MCP; otherwise require command or url.
        if not any([self.command, self.url]):
            raise ValueError("MCP server requires either 'command' (stdio) or 'url' (HTTP)")
        return self


class AgentDefaults(BaseModel):
    """Global defaults applied to every agent unless overridden."""

    model_config = ConfigDict(extra="forbid")

    model: Optional[str] = None
    llm_profile: str = "fast"
    model_provider: str
    preferred_auth_method: Optional[str] = None
    model_reasoning_effort: Optional[str] = None
    model_reasoning_summary: Optional[str] = None
    request_max_retries: Optional[int] = None
    stream_max_retries: Optional[int] = None
    sandbox_mode: str = "danger-full-access"
    ask_for_approval: str = "never"
    skip_git_repo_check: bool = True
    instructions_dir: str = "instructions"
    project_root: str
    local_codex_bin: str
    working_dir: str = "."
    env: Dict[str, str] = Field(default_factory=dict)


class AgentSpec(BaseModel):
    """Single agent entry from agents_config.toml."""

    model_config = ConfigDict(extra="forbid")

    name: str
    agent_folder: str
    instructions_file: str
    model: Optional[str] = None
    llm_profile: Optional[str] = None
    model_provider: Optional[str] = None
    preferred_auth_method: Optional[str] = None
    model_reasoning_effort: Optional[str] = None
    model_reasoning_summary: Optional[str] = None
    request_max_retries: Optional[int] = None
    stream_max_retries: Optional[int] = None
    sandbox_mode: Optional[str] = None
    ask_for_approval: Optional[str] = None
    skip_git_repo_check: Optional[bool] = None
    working_dir: Optional[str] = None
    env: Dict[str, str] = Field(default_factory=dict)
    mcp_servers: Dict[str, MCPServerConfig] = Field(default_factory=dict)


class AgentsConfig(BaseModel):
    """Root config parsed from agents_config.toml."""

    model_config = ConfigDict(extra="forbid")

    defaults: AgentDefaults
    agents: List[AgentSpec]

    @classmethod
    def load(cls, path: Path) -> "AgentsConfig":
        raw = tomllib.loads(Path(path).read_text())
        expanded = _expand_env(raw)
        return cls(**expanded)


class ResolvedAgentConfig(BaseModel):
    """Fully merged agent configuration with resolved paths."""

    model_config = ConfigDict(extra="forbid")

    name: str
    agent_folder: Path
    instructions_path: Path
    model: str
    llm_profile: str
    model_provider: str
    preferred_auth_method: Optional[str] = None
    model_reasoning_effort: Optional[str] = None
    model_reasoning_summary: Optional[str] = None
    request_max_retries: Optional[int] = None
    stream_max_retries: Optional[int] = None
    sandbox_mode: str
    ask_for_approval: str
    skip_git_repo_check: bool
    project_root: Path
    local_codex_bin: Path
    working_dir: Path
    env: Dict[str, str] = Field(default_factory=dict)
    mcp_servers: Dict[str, MCPServerConfig] = Field(default_factory=dict)

    @classmethod
    def from_spec(
        cls,
        *,
        snapshot_root: Path,
        instructions_root: Path,
        defaults: AgentDefaults,
        spec: AgentSpec,
    ) -> "ResolvedAgentConfig":
        from backend.module.config_handler import settings  # lazy import to avoid cycles

        agent_root = (snapshot_root / spec.agent_folder).resolve()
        instructions_path = (instructions_root / spec.instructions_file).resolve()

        merged_env = {**defaults.env, **spec.env}
        working_dir = spec.working_dir or defaults.working_dir

        provider = spec.model_provider or defaults.model_provider
        profile = (spec.llm_profile or defaults.llm_profile).lower()
        resolved_model = spec.model or defaults.model or settings.get_llm_model(profile, provider)

        return cls(
            name=spec.name,
            agent_folder=agent_root,
            instructions_path=instructions_path,
            model=resolved_model,
            llm_profile=profile,
            model_provider=provider,
            preferred_auth_method=spec.preferred_auth_method or defaults.preferred_auth_method,
            model_reasoning_effort=spec.model_reasoning_effort or defaults.model_reasoning_effort,
            model_reasoning_summary=spec.model_reasoning_summary or defaults.model_reasoning_summary,
            request_max_retries=spec.request_max_retries or defaults.request_max_retries,
            stream_max_retries=spec.stream_max_retries or defaults.stream_max_retries,
            sandbox_mode=spec.sandbox_mode or defaults.sandbox_mode,
            ask_for_approval=spec.ask_for_approval or defaults.ask_for_approval,
            skip_git_repo_check=spec.skip_git_repo_check if spec.skip_git_repo_check is not None else defaults.skip_git_repo_check,
            project_root=Path(defaults.project_root).expanduser().resolve(),
            local_codex_bin=Path(defaults.local_codex_bin).expanduser().resolve(),
            working_dir=(agent_root / working_dir).resolve(),
            env=merged_env,
            mcp_servers=spec.mcp_servers,
        )
