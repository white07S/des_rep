from __future__ import annotations

import asyncio
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List, Optional

from backend.module.agents.assembler import assemble_agents
from backend.module.agents.agent_types import ResolvedAgentConfig
from backend.module.logging import LoggingInterceptor

logger = LoggingInterceptor("agents_sessions")


@dataclass
class ThreadOpts:
    working_directory: str
    sandbox_mode: str
    ask_for_approval: str
    skip_git_repo_check: bool
    output_schema: Optional[Dict[str, Any]] = None  # JSON Schema for structured output


class CodexThread:
    """
    Thin wrapper around `codex exec --json` supporting resumable sessions.
    """

    def __init__(
        self,
        *,
        env: Dict[str, str],
        opts: ThreadOpts,
        codex_bin: Path,
        thread_id: Optional[str] = None,
    ):
        self._base_env = env
        self._opts = opts
        self._thread_id = thread_id
        self._schema_file: Optional[str] = None
        self._codex_bin = codex_bin

        if not (self._codex_bin.exists() and os.access(self._codex_bin, os.X_OK)):
            raise FileNotFoundError(
                f"Codex binary not found or not executable at {self._codex_bin}"
            )

        if self._opts.output_schema:
            fd, path = tempfile.mkstemp(suffix=".json", prefix="codex_schema_")
            with os.fdopen(fd, "w") as f:
                json.dump(self._opts.output_schema, f, indent=2)
            self._schema_file = path

    def __del__(self) -> None:
        if self._schema_file and os.path.exists(self._schema_file):
            try:
                os.unlink(self._schema_file)
            except Exception:
                pass

    async def run_streamed(self, prompt: str) -> AsyncIterator[Dict[str, Any]]:
        global_prefix = [
            str(self._codex_bin),
            "--ask-for-approval",
            self._opts.ask_for_approval,
        ]

        if self._thread_id is None:
            cmd = global_prefix + ["exec"]
        else:
            cmd = global_prefix + ["exec", "resume", self._thread_id]

        cmd += [
            "--json",
            "--cd",
            self._opts.working_directory,
            "--sandbox",
            self._opts.sandbox_mode,
        ]
        if self._opts.skip_git_repo_check:
            cmd.append("--skip-git-repo-check")

        if self._schema_file:
            cmd += ["--output-schema", self._schema_file]

        cmd.append(prompt)

        proc_env = os.environ.copy()
        proc_env.update(self._base_env)

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=proc_env,
        )

        stderr_lines: List[str] = []

        async def _drain_stderr() -> None:
            assert proc.stderr is not None
            while True:
                line = await proc.stderr.readline()
                if not line:
                    return
                s = line.decode("utf-8", errors="replace").rstrip("\n")
                stderr_lines.append(s)
                print(f"[codex stderr] {s}")

        stderr_task = asyncio.create_task(_drain_stderr())

        try:
            assert proc.stdout is not None
            buffer = b""

            while True:
                chunk = await proc.stdout.read(64 * 1024)
                if not chunk:
                    if buffer:
                        s = buffer.decode("utf-8", errors="replace").strip()
                        if s:
                            try:
                                event = json.loads(s)
                                if event.get("type") == "thread.started" and event.get("thread_id"):
                                    self._thread_id = event["thread_id"]
                                yield event
                            except json.JSONDecodeError:
                                pass
                    break

                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    s = line.decode("utf-8", errors="replace").strip()
                    if not s:
                        continue
                    try:
                        event = json.loads(s)
                        if event.get("type") == "thread.started" and event.get("thread_id"):
                            self._thread_id = event["thread_id"]
                        yield event
                    except json.JSONDecodeError:
                        pass

            rc = await proc.wait()
            if rc != 0:
                tail = "\n".join(stderr_lines[-80:])
                raise RuntimeError(
                    f"codex exited with code {rc}\n\n"
                    f"Command: {' '.join(cmd)}\n\n"
                    f"--- codex stderr (tail) ---\n{tail}\n"
                )
        finally:
            stderr_task.cancel()


def _find_agent(agent_name: str, config_path: Optional[Path] = None) -> ResolvedAgentConfig:
    agents = assemble_agents(config_path=config_path)
    for agent in agents:
        if agent.name == agent_name:
            return agent
    raise ValueError(f"Agent '{agent_name}' not found in agents_config.toml")


def _build_env(agent: ResolvedAgentConfig) -> Dict[str, str]:
    from backend.module.config_handler import settings

    env: Dict[str, str] = dict(os.environ)
    env.update(agent.env)

    provider = agent.model_provider.lower()
    if provider == "azure":
        env.setdefault("AZURE_OPENAI_LLM_API_KEY", settings.azure.llm.api_key)
        env.setdefault("AZURE_OPENAI_API_KEY", settings.azure.llm.api_key)
        env.setdefault("AZURE_OPENAI_ENDPOINT", str(settings.azure.api_base))
    elif provider == "openai":
        env.setdefault("OPENAI_API_KEY", settings.openai.api_key)
        env.setdefault("OPENAI_API_BASE", str(settings.openai.api_base))

    env["CODEX_HOME"] = str(agent.agent_folder)
    # Align HOME/USER with agent folder per project instruction.
    env["HOME"] = str(agent.agent_folder)
    env["USER"] = env.get("USER") or Path(agent.agent_folder).name
    return env


def _build_thread(agent: ResolvedAgentConfig, thread_id: Optional[str], output_schema: Optional[Dict[str, Any]]) -> CodexThread:
    opts = ThreadOpts(
        working_directory=str(agent.working_dir),
        sandbox_mode=agent.sandbox_mode,
        ask_for_approval=agent.ask_for_approval,
        skip_git_repo_check=agent.skip_git_repo_check,
        output_schema=output_schema,
    )
    env = _build_env(agent)
    return CodexThread(env=env, opts=opts, codex_bin=agent.local_codex_bin, thread_id=thread_id)


async def create_new_session(
    agent_name: str,
    prompt: str,
    *,
    output_schema: Optional[Dict[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Start a new Codex exec session for the given agent."""
    agent = _find_agent(agent_name, config_path=config_path)
    thread = _build_thread(agent, thread_id=None, output_schema=output_schema)
    async for event in thread.run_streamed(prompt):
        yield event


async def resume_session(
    agent_name: str,
    session_id: str,
    prompt: str,
    *,
    output_schema: Optional[Dict[str, Any]] = None,
    config_path: Optional[Path] = None,
) -> AsyncIterator[Dict[str, Any]]:
    """Resume an existing Codex exec session."""
    agent = _find_agent(agent_name, config_path=config_path)
    thread = _build_thread(agent, thread_id=session_id, output_schema=output_schema)
    async for event in thread.run_streamed(prompt):
        yield event


__all__ = ["create_new_session", "resume_session", "CodexThread", "ThreadOpts"]
