from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from saq.job import Job

from backend.module.config_handler import settings
from backend.module.file_handlers import storage
from backend.module.file_handlers.metadata_store import (
    FileRecord,
    JobRecord,
    KnowledgeRecord,
    create_metadata_store,
)
from backend.module.file_handlers.validators import (
    UploadRequest,
    UploadValidationResult,
    UploadType,
    compute_upload_hash,
    prepare_upload,
)
from backend.module.file_handlers.worker import get_queue
from backend.module.logging import LoggingInterceptor
from backend.module.pipelines.csv.csv_pipeline import run_csv_pipeline_and_save
from backend.module.pipelines.csv.csv_pipeline_models import CSVConversionConfig
from backend.module.pipelines.db.schema_assembler import assemble_schema
from backend.module.pipelines.docs.document_pipeline_utils import compute_file_hash
from backend.module.pipelines.docs.document_post_ingest import run_document_post_ingest
from backend.module.pipelines.docs.document_post_pipeline import run_document_post_pipeline
from backend.module.pipelines.docs.document_pre_pipeline import run_document_pre_pipeline
from backend.module.pipelines.knowledge import KnowledgeEntry, ingest_knowledge_entry

logger = LoggingInterceptor("file_handlers_dispatchers")
metadata_store = create_metadata_store(settings.storage.data_root)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _job_id_from_ctx(ctx: Dict[str, Any], fallback: str) -> str:
    job = ctx.get("job")
    if isinstance(job, Job):
        return job.id
    if isinstance(job, dict) and "id" in job:
        return str(job["id"])
    return fallback


def _record_job_state(
    *,
    job_id: str,
    job_type: str,
    state: str,
    user: Optional[str],
    file_id: Optional[str],
    error: Optional[str] = None,
    started_at: Optional[datetime] = None,
    finished_at: Optional[datetime] = None,
    artifact_path: Optional[Path] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    record = JobRecord(
        job_id=job_id,
        job_type=job_type,
        state=state,
        user=user,
        file_id=file_id,
        error=error,
        created_at=_utc_now(),
        updated_at=_utc_now(),
        started_at=started_at,
        finished_at=finished_at,
        artifact_path=artifact_path,
        extra=extra or {},
    )
    metadata_store.upsert_job(record)


def stage_upload(request: UploadRequest) -> UploadValidationResult:
    """Validate and store an upload into originals/, recording metadata."""
    file_hash = compute_upload_hash(request.source_path)
    existing = metadata_store.get_file_by_hash(
        user=request.user,
        file_hash=file_hash,
        file_type=request.upload_type.value,
    )
    if existing:
        merged_metadata = {**(existing.extra or {}), **(request.metadata or {})}
        logger.info(
            "Upload deduplicated by hash",
            file_id=existing.file_id,
            user=request.user,
            file_type=request.upload_type.value,
        )
        refreshed = existing.model_copy(
            update={
                "business_context": request.business_context,
                "extra": merged_metadata,
                "updated_at": _utc_now(),
            }
        )
        metadata_store.upsert_file(refreshed)
        schema_candidate = (
            storage.resolve_original_path(request.user, existing.file_id, "schema.sql")
            if request.upload_type is UploadType.SQLITE
            else None
        )
        return UploadValidationResult(
            file_id=existing.file_id,
            file_hash=file_hash,
            original_path=existing.original_path,
            artifact_dir=existing.artifact_path,
            original_filename=existing.original_filename,
            file_size=existing.file_size,
            schema_path=schema_candidate if schema_candidate and schema_candidate.exists() else None,
        )

    result = prepare_upload(request, precomputed_hash=file_hash)
    record = FileRecord(
        file_id=result.file_id,
        user=request.user,
        business_context=request.business_context,
        file_type=request.upload_type.value,
        original_filename=result.original_filename,
        original_path=result.original_path,
        artifact_path=result.artifact_dir,
        file_hash=result.file_hash,
        file_size=result.file_size,
        created_at=_utc_now(),
        updated_at=_utc_now(),
        extra=request.metadata,
    )
    metadata_store.upsert_file(record)
    logger.info(
        "Upload staged",
        file_id=result.file_id,
        user=request.user,
        file_type=request.upload_type.value,
        original=str(result.original_path),
    )
    return result


async def enqueue_pdf_job(
    file_id: str,
    user: str,
    business_context: str,
    collection_name: str,
    *,
    provider: Optional[str] = None,
    model_profile: Optional[str] = None,
) -> str:
    queue = get_queue()
    job = await queue.enqueue(
        "process_pdf",
        key=queue.job_id(f"process_pdf:{file_id}"),
        file_id=file_id,
        user=user,
        business_context=business_context,
        collection_name=collection_name,
        provider=provider,
        model_profile=model_profile,
        timeout=900,
        retries=2,
        retry_delay=15,
    )
    _record_job_state(
        job_id=job.id,
        job_type="process_pdf",
        state="queued",
        user=user,
        file_id=file_id,
        extra={"collection_name": collection_name},
    )
    return job.id


async def enqueue_csv_job(
    file_id: str,
    user: str,
    business_context: str,
) -> str:
    queue = get_queue()
    job = await queue.enqueue(
        "process_csv",
        key=queue.job_id(f"process_csv:{file_id}"),
        file_id=file_id,
        user=user,
        business_context=business_context,
        timeout=300,
        retries=1,
        retry_delay=10,
    )
    _record_job_state(job_id=job.id, job_type="process_csv", state="queued", user=user, file_id=file_id)
    return job.id


async def enqueue_sqlite_job(
    file_id: str,
    user: str,
    business_context: str,
    *,
    provider: Optional[str] = None,
    model_profile: Optional[str] = None,
    sample_limit: int = 3,
) -> str:
    queue = get_queue()
    job = await queue.enqueue(
        "process_sqlite",
        key=queue.job_id(f"process_sqlite:{file_id}"),
        file_id=file_id,
        user=user,
        business_context=business_context,
        provider=provider,
        model_profile=model_profile,
        sample_limit=sample_limit,
        timeout=600,
        retries=1,
        retry_delay=20,
    )
    _record_job_state(job_id=job.id, job_type="process_sqlite", state="queued", user=user, file_id=file_id)
    return job.id


async def enqueue_knowledge_job(
    user: str,
    collection_name: str,
    query: str,
    answer: str,
    *,
    file_id: Optional[str] = None,
    force: bool = False,
    provider: Optional[str] = None,
) -> str:
    queue = get_queue()
    job = await queue.enqueue(
        "process_knowledge",
        key=queue.job_id(f"process_knowledge:{collection_name}:{query[:16]}"),
        user=user,
        collection_name=collection_name,
        query=query,
        answer=answer,
        file_id=file_id,
        force=force,
        provider=provider,
        timeout=180,
        retries=1,
        retry_delay=10,
    )
    _record_job_state(
        job_id=job.id,
        job_type="process_knowledge",
        state="queued",
        user=user,
        file_id=file_id,
        extra={"collection_name": collection_name},
    )
    return job.id


async def process_pdf(
    ctx: Dict[str, Any],
    *,
    file_id: str,
    user: str,
    business_context: str,
    collection_name: str,
    provider: Optional[str] = None,
    model_profile: Optional[str] = None,
) -> Dict[str, Any]:
    job_id = _job_id_from_ctx(ctx, f"process_pdf:{file_id}")
    artifact_dir = storage.ensure_directory(storage.get_artifacts_dir(user, file_id))
    original_pdf = storage.get_original_file_path(user, file_id, ".pdf")

    pre_json_path = storage.resolve_artifact_path(user, file_id, "pre.json")
    pre_md_path = storage.resolve_artifact_path(user, file_id, "pre.md")
    post_md_path = storage.resolve_artifact_path(user, file_id, "post.md")
    images_dir = storage.resolve_artifact_path(user, file_id, "images")
    chunk_mapping_path = storage.resolve_artifact_path(user, file_id, "chunk_mapping.sqlite")
    qdrant_path = storage.resolve_artifact_path(user, file_id, "qdrant")
    pre_markdown: Optional[str] = None
    processed_markdown: Optional[str] = None
    file_hash = await asyncio.to_thread(compute_file_hash, original_pdf)

    if post_md_path.exists() and chunk_mapping_path.exists() and qdrant_path.exists():
        logger.info("PDF artifacts already present; skipping processing", job_id=job_id, file_id=file_id)
        _record_job_state(
            job_id=job_id,
            job_type="process_pdf",
            state="complete",
            user=user,
            file_id=file_id,
            started_at=_utc_now(),
            finished_at=_utc_now(),
            artifact_path=post_md_path,
            extra={"dedup": True},
        )
        return {
            "post_markdown": post_md_path,
            "qdrant_path": qdrant_path,
            "chunk_mapping": chunk_mapping_path,
        }

    _record_job_state(
        job_id=job_id,
        job_type="process_pdf",
        state="active",
        user=user,
        file_id=file_id,
        started_at=_utc_now(),
        extra={"collection_name": collection_name},
    )
    logger.info("Processing PDF", job_id=job_id, file_id=file_id, original=str(original_pdf))

    try:
        # Pre-pipeline
        if pre_md_path.exists() and pre_json_path.exists():
            pre_markdown = await asyncio.to_thread(pre_md_path.read_text, encoding="utf-8")
            logger.info("Pre artifacts found; skipping pre-pipeline", job_id=job_id)
        else:
            pre_result = await run_document_pre_pipeline(original_pdf, image_path_prefix="images")
            pre_markdown = pre_result.markdown
            await asyncio.to_thread(storage.atomic_write_text, pre_md_path, pre_result.markdown)
            await asyncio.to_thread(
                storage.atomic_write_text,
                pre_json_path,
                json.dumps(pre_result.analysis, ensure_ascii=False, indent=2),
            )
            for image in pre_result.images:
                target = storage.resolve_artifact_path(user, file_id, image.suggested_path)
                await asyncio.to_thread(storage.atomic_write_bytes, target, image.content_bytes)
            logger.info("Pre-pipeline completed", job_id=job_id, images=len(pre_result.images))

        # Post-pipeline
        if post_md_path.exists():
            processed_markdown = await asyncio.to_thread(post_md_path.read_text, encoding="utf-8")
            logger.info("Post artifact found; skipping post-pipeline", job_id=job_id)
        else:
            post_result = await run_document_post_pipeline(
                pre_markdown,
                provider=provider,
                model_profile=model_profile,
            )
            processed_markdown = post_result.processed_markdown
            await asyncio.to_thread(storage.atomic_write_text, post_md_path, processed_markdown)
            logger.info("Post-pipeline completed", job_id=job_id, replacements=post_result.total_items)

        ingest_result = await run_document_post_ingest(
            processed_markdown,
            file_hash=file_hash,
            collection_name=collection_name,
            qdrant_path=qdrant_path,
            chunk_mapping_path=chunk_mapping_path,
            provider=provider,
        )
        _record_job_state(
            job_id=job_id,
            job_type="process_pdf",
            state="complete",
            user=user,
            file_id=file_id,
            finished_at=_utc_now(),
            artifact_path=post_md_path,
        )
        logger.info(
            "PDF processing completed",
            job_id=job_id,
            inserted=ingest_result.inserted_chunks,
            reused=ingest_result.reused_chunks,
        )
        return {
            "post_markdown": post_md_path,
            "qdrant_path": qdrant_path,
            "chunk_mapping": chunk_mapping_path,
        }
    except Exception as exc:  # noqa: BLE001
        _record_job_state(
            job_id=job_id,
            job_type="process_pdf",
            state="failed",
            user=user,
            file_id=file_id,
            finished_at=_utc_now(),
            error=str(exc),
        )
        logger.exception("PDF processing failed", job_id=job_id, file_id=file_id, error=str(exc))
        raise

async def process_csv(
    ctx: Dict[str, Any],
    *,
    file_id: str,
    user: str,
    business_context: str,
) -> Dict[str, Any]:
    job_id = _job_id_from_ctx(ctx, f"process_csv:{file_id}")
    original_csv = storage.get_original_file_path(user, file_id, ".csv")
    parquet_path = storage.resolve_artifact_path(user, file_id, "data.parquet")
    if parquet_path.exists():
        logger.info("Parquet already present; skipping pipeline", job_id=job_id, output=str(parquet_path))
        _record_job_state(
            job_id=job_id,
            job_type="process_csv",
            state="complete",
            user=user,
            file_id=file_id,
            started_at=_utc_now(),
            finished_at=_utc_now(),
            artifact_path=parquet_path,
            extra={"dedup": True},
        )
        return {"parquet_path": parquet_path}

    _record_job_state(
        job_id=job_id,
        job_type="process_csv",
        state="active",
        user=user,
        file_id=file_id,
        started_at=_utc_now(),
    )
    logger.info("Processing CSV", job_id=job_id, file_id=file_id)

    try:
        config = CSVConversionConfig(input_path=original_csv, output_path=parquet_path)
        result = await run_csv_pipeline_and_save(config)
        logger.info("CSV pipeline complete", job_id=job_id, rows=result.row_count)
        _record_job_state(
            job_id=job_id,
            job_type="process_csv",
            state="complete",
            user=user,
            file_id=file_id,
            finished_at=_utc_now(),
            artifact_path=parquet_path,
        )
        return {"parquet_path": parquet_path}
    except Exception as exc:  # noqa: BLE001
        _record_job_state(
            job_id=job_id,
            job_type="process_csv",
            state="failed",
            user=user,
            file_id=file_id,
            finished_at=_utc_now(),
            error=str(exc),
        )
        logger.exception("CSV processing failed", job_id=job_id, error=str(exc))
        raise

async def process_sqlite(
    ctx: Dict[str, Any],
    *,
    file_id: str,
    user: str,
    business_context: str,
    provider: Optional[str] = None,
    model_profile: Optional[str] = None,
    sample_limit: int = 3,
) -> Dict[str, Any]:
    job_id = _job_id_from_ctx(ctx, f"process_sqlite:{file_id}")
    sqlite_path = storage.get_original_file_path(user, file_id, ".sqlite")
    schema_path = storage.resolve_original_path(user, file_id, "schema.sql")
    output_spec = storage.resolve_artifact_path(user, file_id, "ai_spec_schema.json")
    artifact_sqlite = storage.resolve_artifact_path(user, file_id, "source.sqlite")
    artifact_schema = storage.resolve_artifact_path(user, file_id, "schema.sql")

    def _ensure_original_artifacts() -> None:
        if not artifact_sqlite.exists():
            storage.atomic_copy(sqlite_path, artifact_sqlite)
        if schema_path.exists() and not artifact_schema.exists():
            storage.atomic_copy(schema_path, artifact_schema)

    _ensure_original_artifacts()

    if output_spec.exists():
        logger.info("Schema spec already exists; skipping assembly", job_id=job_id, output=str(output_spec))
        _record_job_state(
            job_id=job_id,
            job_type="process_sqlite",
            state="complete",
            user=user,
            file_id=file_id,
            started_at=_utc_now(),
            finished_at=_utc_now(),
            artifact_path=output_spec,
            extra={"dedup": True},
        )
        return {"schema_path": output_spec}

    _record_job_state(
        job_id=job_id,
        job_type="process_sqlite",
        state="active",
        user=user,
        file_id=file_id,
        started_at=_utc_now(),
    )
    logger.info("Processing SQLite schema", job_id=job_id, file_id=file_id)

    try:
        _ensure_original_artifacts()
        await assemble_schema(
            sqlite_path=sqlite_path,
            schema_sql_path=schema_path,
            output_path=output_spec,
            database_id=file_id,
            provider=provider,
            model_profile=model_profile,
            sample_limit=sample_limit,
            business_context=business_context,
        )
        _record_job_state(
            job_id=job_id,
            job_type="process_sqlite",
            state="complete",
            user=user,
            file_id=file_id,
            finished_at=_utc_now(),
            artifact_path=output_spec,
        )
        return {"schema_path": output_spec}
    except Exception as exc:  # noqa: BLE001
        _record_job_state(
            job_id=job_id,
            job_type="process_sqlite",
            state="failed",
            user=user,
            file_id=file_id,
            finished_at=_utc_now(),
            error=str(exc),
        )
        logger.exception("SQLite processing failed", job_id=job_id, error=str(exc))
        raise

async def process_knowledge(
    ctx: Dict[str, Any],
    *,
    user: str,
    collection_name: str,
    query: str,
    answer: str,
    file_id: Optional[str] = None,
    force: bool = False,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    job_id = _job_id_from_ctx(ctx, f"process_knowledge:{collection_name}:{query[:8]}")
    _record_job_state(
        job_id=job_id,
        job_type="process_knowledge",
        state="active",
        user=user,
        file_id=file_id,
        started_at=_utc_now(),
        extra={"collection_name": collection_name},
    )

    artifact_dir = storage.ensure_directory(storage.get_artifacts_dir(user, file_id or "knowledge"))
    qdrant_path = artifact_dir / "knowledge_qdrant"
    entry_id = f"{collection_name}-{uuid_fragment()}"
    output_path = artifact_dir / "knowledge" / f"{entry_id}.json"

    try:
        entry = KnowledgeEntry(
            collection_name=collection_name,
            query=query,
            answer=answer,
            user=user,
            file_id=file_id,
        )
        result = await ingest_knowledge_entry(
            entry,
            qdrant_path=qdrant_path,
            output_path=output_path,
            similarity_threshold=0.95,
            force=force,
            provider=provider,
        )
        metadata_store.upsert_knowledge(
            KnowledgeRecord(
                knowledge_id=entry_id,
                user=user,
                collection_name=collection_name,
                query=query,
                answer_path=result.output_path,
                similarity=result.similarity,
                forced=force,
                file_id=file_id,
                created_at=_utc_now(),
                extra={"reused": result.reused},
            )
        )
        _record_job_state(
            job_id=job_id,
            job_type="process_knowledge",
            state="complete",
            user=user,
            file_id=file_id,
            finished_at=_utc_now(),
            artifact_path=result.output_path,
        )
        logger.info("Knowledge ingestion complete", job_id=job_id, collection=collection_name)
        return result.model_dump()
    except Exception as exc:  # noqa: BLE001
        _record_job_state(
            job_id=job_id,
            job_type="process_knowledge",
            state="failed",
            user=user,
            file_id=file_id,
            finished_at=_utc_now(),
            error=str(exc),
        )
        logger.exception("Knowledge processing failed", job_id=job_id, error=str(exc))
        raise

def uuid_fragment() -> str:
    from uuid import uuid4

    return uuid4().hex[:12]


__all__ = [
    "stage_upload",
    "enqueue_pdf_job",
    "enqueue_csv_job",
    "enqueue_sqlite_job",
    "enqueue_knowledge_job",
    "process_pdf",
    "process_csv",
    "process_sqlite",
    "process_knowledge",
]
