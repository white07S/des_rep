from __future__ import annotations

import argparse
import asyncio
import base64
import json
from pathlib import Path
from typing import List, Optional

from openai.types.responses import response_text_delta_event, response_text_done_event
from pydantic import BaseModel, ConfigDict

from backend.module.config_handler import settings
from backend.module.logging import LoggingInterceptor
from backend.module.providers.document_intelligence import get_document_intelligence_client
from backend.module.providers.embeddings import get_embeddings_client_bundle
from backend.module.providers.llm import get_llm_client_bundle

logger = LoggingInterceptor("providers_tests")


async def test_llm_stream(provider: Optional[str] = None) -> None:
    bundle = get_llm_client_bundle(provider=provider)
    client, model = bundle.client, bundle.model
    logger.info("Starting LLM stream test", provider=provider or settings.default_provider, model=model)
    print(f"[LLM stream] provider={provider or settings.default_provider} model={model}")

    resp = await client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": "Say hi"}]}],
        stream=True,
    )

    text_chunks: List[str] = []
    async for event in resp:
        if isinstance(event, response_text_delta_event.ResponseTextDeltaEvent):
            text_chunks.append(event.delta)
        elif isinstance(event, response_text_done_event.ResponseTextDoneEvent):
            text_chunks.append(event.text)
    output_text = "".join(text_chunks)
    logger.info("Stream collected text", output=output_text)
    print(f"[LLM stream] output={output_text}")


async def test_llm_structured(provider: Optional[str] = None) -> None:
    bundle = get_llm_client_bundle(provider=provider)
    client, model = bundle.client, bundle.model
    logger.info("Starting LLM structured test", provider=provider or settings.default_provider, model=model)
    print(f"[LLM structured] provider={provider or settings.default_provider} model={model}")

    class OutputSchema(BaseModel):
        model_config = ConfigDict(
            extra="forbid",
            json_schema_extra={"additionalProperties": False},
        )
        color: str
        number: int

    try:
        parsed_response = await client.responses.parse(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Return a JSON object with keys color and number."}],
                }
            ],
            text_format=OutputSchema,
            prompt_cache_key="test-structured"
        )
        parsed_obj = parsed_response.output_parsed
        parsed_dict = parsed_obj.model_dump() if parsed_obj else {}
        logger.info("Structured output", parsed=parsed_dict)
        print(f"[LLM structured] output={parsed_dict}")
    except Exception as exc:  # noqa: BLE001
        logger.error("Structured parse failed", error=str(exc))
        print(f"[LLM structured] error={exc}")


async def test_llm_image(provider: Optional[str] = None, image_path: Optional[Path] = None) -> None:
    bundle = get_llm_client_bundle(provider=provider)
    client, model = bundle.client, bundle.model
    img_path = image_path or settings.get_testing_file_path("model.png")
    logger.info("Starting LLM image test", provider=provider or settings.default_provider, model=model, image=str(img_path))
    print(f"[LLM image] provider={provider or settings.default_provider} model={model} image={img_path}")

    with img_path.open("rb") as f:
        image_bytes = f.read()
    image_b64 = base64.b64encode(image_bytes).decode("ascii")

    resp = await client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "Describe this image succinctly."},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{image_b64}", "detail": "auto"},
                ],
            }
        ],
    )
    logger.info("Image response received", output=resp.output_text)
    print(f"[LLM image] output={resp.output_text}")


async def test_llm_batch(provider: Optional[str] = None) -> None:
    bundle = get_llm_client_bundle(provider=provider)
    client, model = bundle.client, bundle.model
    logger.info("Starting LLM batch test", provider=provider or settings.default_provider, model=model)
    print(f"[LLM batch] provider={provider or settings.default_provider} model={model}")

    prompts = [
        {"role": "user", "content": [{"type": "input_text", "text": "1+1="}]},
        {"role": "user", "content": [{"type": "input_text", "text": "Say a greeting"}]},
    ]
    # Simulate batch by sending multiple items and reading outputs list
    resp = await client.responses.create(
        model=model,
        input=prompts,
        stream=False,
        prompt_cache_key="test-batch",
    )
    logger.info("Batch outputs", outputs=resp.output_text)
    print(f"[LLM batch] outputs={resp.output_text}")


async def test_embeddings(provider: Optional[str] = None) -> None:
    bundle = get_embeddings_client_bundle(provider=provider)
    client, model = bundle.client, bundle.model
    logger.info("Starting embeddings test", provider=provider or settings.default_provider, model=model)
    print(f"[Embeddings] provider={provider or settings.default_provider} model={model}")

    single = await client.embeddings.create(model=model, input="hello world")
    batch = await client.embeddings.create(model=model, input=["hello", "world"])
    logger.info("Embeddings lengths", single_len=len(single.data[0].embedding), batch_count=len(batch.data))
    print(f"[Embeddings] single_dim={len(single.data[0].embedding)} batch_count={len(batch.data)}")


async def test_document_intelligence(file_path: Optional[Path] = None) -> None:
    pdf_path = file_path or settings.get_testing_file_path("ada.pdf")
    logger.info("Starting document intelligence test", file=str(pdf_path))
    print(f"[DocInt] file={pdf_path}")

    client = get_document_intelligence_client()
    async with client:
        with pdf_path.open("rb") as f:
            poller = await client.begin_analyze_document("prebuilt-layout", f)
            result = await poller.result()
        pages = len(result.pages)
        paragraphs = len(result.paragraphs or [])
        logger.info("Document intelligence result", pages=pages, paragraphs=paragraphs)
        print(f"[DocInt] pages={pages} paragraphs={paragraphs}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run provider smoke tests.")
    parser.add_argument("--provider", choices=["openai", "azure"], default=settings.default_provider)
    args = parser.parse_args()

    provider = args.provider
    await test_llm_stream(provider)
    await test_llm_structured(provider)
    await test_llm_image(provider)
    await test_llm_batch(provider)
    await test_embeddings(provider)
    await test_document_intelligence()


if __name__ == "__main__":
    asyncio.run(main())
