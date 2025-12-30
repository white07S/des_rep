from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

from openai import AsyncAzureOpenAI, AsyncOpenAI

from backend.module.config_handler import settings
from backend.module.logging import LoggingInterceptor
from backend.module.providers.llm import DEFAULT_HEADERS

logger = LoggingInterceptor("providers_embeddings")

EmbeddingsClient = Union[AsyncOpenAI, AsyncAzureOpenAI]


@dataclass
class EmbeddingsClientBundle:
    client: EmbeddingsClient
    model: str


def get_embeddings_client(provider: Optional[str] = None) -> EmbeddingsClient:
    provider_name = settings.resolve_provider(provider)
    if provider_name == "openai":
        logger.info("Creating OpenAI embeddings client")
        return AsyncOpenAI(
            api_key=settings.openai.api_key,
            base_url=settings.openai.api_base,
            default_headers=DEFAULT_HEADERS,
        )

    logger.info("Creating Azure embeddings client")
    return AsyncAzureOpenAI(
        api_key=settings.azure.embeddings.api_key,
        azure_endpoint=settings.azure.api_base,
        api_version=settings.azure.embeddings.api_version,
        default_headers=DEFAULT_HEADERS,
    )


def get_embeddings_model(provider: Optional[str] = None) -> str:
    return settings.get_embeddings_model(provider)


def get_embeddings_client_bundle(provider: Optional[str] = None) -> EmbeddingsClientBundle:
    client = get_embeddings_client(provider)
    model = get_embeddings_model(provider)
    return EmbeddingsClientBundle(client=client, model=model)

