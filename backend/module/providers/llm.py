from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Union

from openai import AsyncAzureOpenAI, AsyncOpenAI

from backend.module.config_handler import settings
from backend.module.logging import LoggingInterceptor

PROMPT_CACHING_BETA = "prompt-caching=v1"
DEFAULT_HEADERS: Dict[str, str] = {"OpenAI-Beta": PROMPT_CACHING_BETA}

logger = LoggingInterceptor("providers_llm")


LLMClient = Union[AsyncOpenAI, AsyncAzureOpenAI]


@dataclass
class LLMClientBundle:
    client: LLMClient
    model: str


def get_llm_client(provider: Optional[str] = None) -> LLMClient:
    """Return an async LLM client for the requested provider."""
    provider_name = settings.resolve_provider(provider)
    if provider_name == "openai":
        logger.info("Creating OpenAI LLM client")
        return AsyncOpenAI(
            api_key=settings.openai.api_key,
            base_url=settings.openai.api_base,
            default_headers=DEFAULT_HEADERS,
        )

    logger.info("Creating Azure OpenAI LLM client")
    return AsyncAzureOpenAI(
        api_key=settings.azure.llm.api_key,
        azure_endpoint=settings.azure.api_base,
        api_version=settings.azure.llm.api_version,
        default_headers=DEFAULT_HEADERS,
    )


def get_llm_model(profile: Optional[str] = None, provider: Optional[str] = None) -> str:
    """Resolve the model/deployment for the given profile and provider."""
    return settings.get_llm_model(profile, provider)


def get_llm_client_bundle(
    profile: Optional[str] = None, provider: Optional[str] = None
) -> LLMClientBundle:
    """Convenience helper to fetch both client and resolved model."""
    provider_name = settings.resolve_provider(provider)
    model = get_llm_model(profile, provider_name)
    client = get_llm_client(provider_name)
    return LLMClientBundle(client=client, model=model)


def get_prompt_caching_headers() -> Dict[str, str]:
    """Expose the prompt-caching header for callers that need to augment requests."""
    return DEFAULT_HEADERS
