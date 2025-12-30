from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv

MODULE_DIR = Path(__file__).resolve().parent
ENV_PATH = MODULE_DIR.parent / ".env"
CONFIG_PATH = MODULE_DIR.parent / "config.toml"

ENV_PATTERN = re.compile(r"\$\{env:([A-Z0-9_]+)(?:\|([^}]+))?}")


def _expand_env(value: Any) -> Any:
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


def _require(value: Optional[str], name: str) -> str:
    if value is None or value == "":
        raise ValueError(f"Missing required config value for '{name}'")
    return value


@dataclass
class OpenAILLMSettings:
    default_profile: str
    models: Dict[str, str]


@dataclass
class OpenAISettings:
    api_base: str
    api_key: str
    llm: OpenAILLMSettings
    embeddings_model: str


@dataclass
class AzureLLMSettings:
    api_version: str
    api_key: str
    default_profile: str
    deployments: Dict[str, str]


@dataclass
class AzureEmbeddingsSettings:
    api_version: str
    api_key: str
    deployment: str


@dataclass
class AzureSettings:
    api_base: str
    llm: AzureLLMSettings
    embeddings: AzureEmbeddingsSettings


@dataclass
class DocumentIntelligenceSettings:
    endpoint: str
    api_key: str


@dataclass
class StorageSettings:
    data_root: Path


@dataclass
class Settings:
    default_provider: str
    openai: OpenAISettings
    azure: AzureSettings
    doc_intelligence: DocumentIntelligenceSettings
    storage: StorageSettings
    testing_files_dir: Optional[Path]
    logging_directory: Path

    def resolve_provider(self, override: Optional[str] = None) -> str:
        provider = (override or self.default_provider or "openai").lower()
        if provider not in {"openai", "azure"}:
            raise ValueError(f"Unsupported provider '{provider}'")
        return provider

    def get_llm_model(self, profile: Optional[str] = None, provider: Optional[str] = None) -> str:
        provider_name = self.resolve_provider(provider)
        if provider_name == "openai":
            profile_name = (profile or self.openai.llm.default_profile).lower()
            try:
                return self.openai.llm.models[profile_name]
            except KeyError as exc:
                raise KeyError(f"Missing OpenAI model for profile '{profile_name}'") from exc
        profile_name = (profile or self.azure.llm.default_profile).lower()
        try:
            return self.azure.llm.deployments[profile_name]
        except KeyError as exc:
            raise KeyError(f"Missing Azure deployment for profile '{profile_name}'") from exc

    def get_embeddings_model(self, provider: Optional[str] = None) -> str:
        provider_name = self.resolve_provider(provider)
        if provider_name == "openai":
            return self.openai.embeddings_model
        return self.azure.embeddings.deployment

    def get_testing_file_path(self, filename: str) -> Path:
        if not self.testing_files_dir:
            raise ValueError("Testing directory is not configured.")
        return Path(self.testing_files_dir) / filename


def _load_settings() -> Settings:
    load_dotenv(ENV_PATH)

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Config file not found at {CONFIG_PATH}")

    with CONFIG_PATH.open("rb") as f:
        raw = tomllib.load(f)
    data = _expand_env(raw)

    providers = data.get("providers", {})
    openai_cfg = providers.get("openai", {})
    azure_cfg = providers.get("azure", {})

    openai_models_section = openai_cfg.get("llm", {}).get("models", {})
    openai_models = {
        k.lower(): v.get("model")
        for k, v in openai_models_section.items()
        if isinstance(v, dict) and v.get("model")
    }

    openai_settings = OpenAISettings(
        api_base=_require(openai_cfg.get("api_base"), "providers.openai.api_base"),
        api_key=_require(openai_cfg.get("api_key"), "providers.openai.api_key"),
        llm=OpenAILLMSettings(
            default_profile=_require(
                openai_cfg.get("llm", {}).get("default_profile"),
                "providers.openai.llm.default_profile",
            ),
            models=openai_models,
        ),
        embeddings_model=_require(
            openai_cfg.get("embeddings", {}).get("model"),
            "providers.openai.embeddings.model",
        ),
    )

    azure_models_section = azure_cfg.get("llm", {}).get("models", {})
    azure_deployments = {
        k.lower(): v.get("deployment")
        for k, v in azure_models_section.items()
        if isinstance(v, dict) and v.get("deployment")
    }

    azure_settings = AzureSettings(
        api_base=_require(azure_cfg.get("api_base"), "providers.azure.api_base"),
        llm=AzureLLMSettings(
            api_version=_require(azure_cfg.get("llm", {}).get("api_version"), "providers.azure.llm.api_version"),
            api_key=_require(azure_cfg.get("llm", {}).get("api_key"), "providers.azure.llm.api_key"),
            default_profile=_require(azure_cfg.get("llm", {}).get("default_profile"), "providers.azure.llm.default_profile"),
            deployments=azure_deployments,
        ),
        embeddings=AzureEmbeddingsSettings(
            api_version=_require(azure_cfg.get("embeddings", {}).get("api_version"), "providers.azure.embeddings.api_version"),
            api_key=_require(azure_cfg.get("embeddings", {}).get("api_key"), "providers.azure.embeddings.api_key"),
            deployment=_require(azure_cfg.get("embeddings", {}).get("deployment"), "providers.azure.embeddings.deployment"),
        ),
    )

    doc_cfg = data.get("azure_document_intelligence", {})
    doc_settings = DocumentIntelligenceSettings(
        endpoint=_require(doc_cfg.get("endpoint"), "azure_document_intelligence.endpoint"),
        api_key=_require(doc_cfg.get("api_key"), "azure_document_intelligence.api_key"),
    )

    storage_cfg = data.get("storage", {})
    storage_settings = StorageSettings(
        data_root=Path(_require(storage_cfg.get("data_root"), "storage.data_root")).expanduser().resolve(),
    )

    testing_dir_raw = data.get("testing_files", {}).get("testing_directory")
    logging_dir_raw = data.get("logging", {}).get("logging_directory")

    return Settings(
        default_provider=(providers.get("default_provider", "openai")),
        openai=openai_settings,
        azure=azure_settings,
        doc_intelligence=doc_settings,
        storage=storage_settings,
        testing_files_dir=Path(testing_dir_raw) if testing_dir_raw else None,
        logging_directory=Path(_require(logging_dir_raw, "logging.logging_directory")),
    )


settings = _load_settings()
