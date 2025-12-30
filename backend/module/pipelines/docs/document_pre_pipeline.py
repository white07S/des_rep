from __future__ import annotations

import asyncio
from pathlib import Path

from backend.module.logging import LoggingInterceptor
from backend.module.pipelines.docs.document_pipeline_models import DocumentPreprocessResult
from backend.module.pipelines.docs.document_pipeline_utils import build_markdown_content, compute_file_hash
from backend.module.providers.document_intelligence import get_document_intelligence_client

logger = LoggingInterceptor("pipelines_docs_pre")


async def run_document_pre_pipeline(pdf_path: Path, *, image_path_prefix: str = "images") -> DocumentPreprocessResult:
    """
    Run the document pre-pipeline: Azure Document Intelligence analysis + markdown/image extraction.

    Returns in-memory content only (no file writes).
    """
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise FileNotFoundError(f"Input must be an existing PDF file: {pdf_path}")

    file_hash = await asyncio.to_thread(compute_file_hash, pdf_path)
    logger.info("Starting pre-pipeline", pdf=str(pdf_path), hash=file_hash)

    client = get_document_intelligence_client()
    try:
        with pdf_path.open("rb") as f:
            poller = await client.begin_analyze_document(
                "prebuilt-layout",
                f,
                content_type="application/pdf",
                output_content_format="markdown",
            )
        result = await poller.result()
        analysis = result.as_dict()
        logger.info("Analysis complete", hash=file_hash)
    finally:
        try:
            await client.close()
        except Exception:
            pass

    markdown, images = await asyncio.to_thread(
        build_markdown_content,
        analysis,
        pdf_path,
        file_hash,
        image_path_prefix=image_path_prefix,
    )
    logger.info(
        "Pre-pipeline finished",
        images=len(images),
        markdown_length=len(markdown),
        analysis_keys=len(analysis),
    )
    return DocumentPreprocessResult(
        file_hash=file_hash,
        original_filename=pdf_path.name,
        analysis=analysis,
        markdown=markdown,
        images=images,
    )


__all__ = ["run_document_pre_pipeline"]
