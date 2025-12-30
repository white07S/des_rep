from __future__ import annotations

from pathlib import Path
from typing import Optional

from backend.module.logging import LoggingInterceptor
from backend.module.pipelines.docs.document_pipeline_models import DocumentPipelineResult
from backend.module.pipelines.docs.document_post_ingest import run_document_post_ingest
from backend.module.pipelines.docs.document_post_pipeline import run_document_post_pipeline
from backend.module.pipelines.docs.document_pre_pipeline import run_document_pre_pipeline

logger = LoggingInterceptor("pipelines_docs_pipeline")


async def run_document_pipeline(
    pdf_path: Path,
    *,
    collection_name: str,
    qdrant_path: Path,
    chunk_mapping_path: Path,
    provider: Optional[str] = None,
    model_profile: Optional[str] = None,
    image_path_prefix: str = "images",
) -> DocumentPipelineResult:
    """Run the full document pipeline (pre + post + ingest) and return in-memory content."""
    logger.info("Running document pipeline", pdf=str(pdf_path))
    pre = await run_document_pre_pipeline(pdf_path, image_path_prefix=image_path_prefix)
    post = await run_document_post_pipeline(
        pre.markdown,
        provider=provider,
        model_profile=model_profile,
    )
    ingest = await run_document_post_ingest(
        post.processed_markdown,
        file_hash=pre.file_hash,
        collection_name=collection_name,
        qdrant_path=qdrant_path,
        chunk_mapping_path=chunk_mapping_path,
        provider=provider,
    )
    logger.info("Document pipeline finished", pdf=str(pdf_path), hash=pre.file_hash)
    return DocumentPipelineResult(pre=pre, post=post, ingest=ingest)


__all__ = ["run_document_pipeline"]
