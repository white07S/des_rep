from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union

from agents import OpenAIResponsesModel, set_default_openai_client
from openai import AsyncAzureOpenAI, AsyncOpenAI

from backend.module.config_handler import settings
from backend.module.logging import LoggingInterceptor
from backend.module.providers.llm import DEFAULT_HEADERS

logger = LoggingInterceptor("providers_oai_agents")

AgentsClient = Union[AsyncOpenAI, AsyncAzureOpenAI]


@dataclass
class AgentsClientBundle:
    model: OpenAIResponsesModel
    client: AgentsClient


def _build_openai_client() -> AsyncOpenAI:
    logger.info("Creating OpenAI Agents client")
    return AsyncOpenAI(
        api_key=settings.openai.api_key,
        base_url=settings.openai.api_base,
        default_headers=DEFAULT_HEADERS,
    )


def _build_azure_client() -> AsyncAzureOpenAI:
    logger.info("Creating Azure Agents client")
    os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")
    return AsyncAzureOpenAI(
        api_key=settings.azure.llm.api_key,
        azure_endpoint=settings.azure.api_base,
        api_version=settings.azure.llm.api_version,
        default_headers=DEFAULT_HEADERS,
    )


def _resolve_model_name(profile: Optional[str], provider: Optional[str]) -> str:
    return settings.get_llm_model(profile, provider)


def create_responses_model(
    profile: Optional[str] = None,
    provider: Optional[str] = None,
    *,
    set_default_client: bool = True,
) -> Tuple[OpenAIResponsesModel, AgentsClient]:
    """
    Build an OpenAIResponsesModel backed by the configured provider.
    Returns the model and the created async OpenAI client (remember to close).
    """
    os.environ.setdefault("OPENAI_AGENTS_DISABLE_TRACING", "1")
    provider_name = settings.resolve_provider(provider)
    client = _build_openai_client() if provider_name == "openai" else _build_azure_client()
    model_name = _resolve_model_name(profile, provider_name)
    model = OpenAIResponsesModel(model=model_name, openai_client=client)

    if set_default_client:
        set_default_openai_client(client)

    logger.info("Responses model ready", provider=provider_name, profile=profile or "default", model=model_name)
    return model, client


async def close_client(client: AgentsClient) -> None:
    """Close the async client returned by create_responses_model."""
    await client.close()
