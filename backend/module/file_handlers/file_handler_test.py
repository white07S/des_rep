from __future__ import annotations

import asyncio
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.module.config_handler import settings
from backend.module.file_handlers.dispatchers import (
    enqueue_csv_job,
    enqueue_knowledge_job,
    enqueue_pdf_job,
    enqueue_sqlite_job,
    process_csv,
    process_knowledge,
    process_pdf,
    process_sqlite,
    stage_upload,
)
from backend.module.file_handlers.validators import UploadRequest, UploadType
from backend.module.file_handlers.worker import get_queue, start_default_worker, stop_worker
from backend.module.logging import LoggingInterceptor

logger = LoggingInterceptor("file_handler_test")

TEST_FILES_DIR = Path(settings.testing_files_dir or Path(__file__).resolve().parents[3] / "test_files")
TEST_USER = "demo_user"
BUSINESS_CONTEXT = "Demo business context"


async def test_pdf() -> None:
    pdf_path = TEST_FILES_DIR / "ada.pdf"
    request = UploadRequest(
        user=TEST_USER,
        business_context=BUSINESS_CONTEXT,
        upload_type=UploadType.PDF,
        source_path=pdf_path,
    )
    upload = stage_upload(request)
    print(f"[pdf] staged upload -> file_id={upload.file_id}, original={upload.original_path}")

    ctx = {"job": {"id": f"test-pdf-{upload.file_id}"}}
    result = await process_pdf(
        ctx,
        file_id=upload.file_id,
        user=request.user,
        business_context=request.business_context,
        collection_name="test_docs",
    )
    print(f"[pdf] completed -> post_markdown={result['post_markdown']}")
    print(f"[pdf] qdrant={result['qdrant_path']} chunk_mapping={result['chunk_mapping']}")


async def test_csv() -> None:
    csv_path = TEST_FILES_DIR / "testing.csv"
    request = UploadRequest(
        user=TEST_USER,
        business_context=BUSINESS_CONTEXT,
        upload_type=UploadType.CSV,
        source_path=csv_path,
    )
    upload = stage_upload(request)
    print(f"[csv] staged upload -> file_id={upload.file_id}, original={upload.original_path}")

    ctx = {"job": {"id": f"test-csv-{upload.file_id}"}}
    result = await process_csv(
        ctx,
        file_id=upload.file_id,
        user=request.user,
        business_context=request.business_context,
    )
    print(f"[csv] completed -> parquet={result['parquet_path']}")


async def test_db() -> None:
    sqlite_path = TEST_FILES_DIR / "department_store.sqlite"
    schema_path = TEST_FILES_DIR / "schema.sql"
    request = UploadRequest(
        user=TEST_USER,
        business_context=BUSINESS_CONTEXT,
        upload_type=UploadType.SQLITE,
        source_path=sqlite_path,
        schema_sql_path=schema_path,
    )
    upload = stage_upload(request)
    print(f"[db] staged upload -> file_id={upload.file_id}, original={upload.original_path}")

    ctx = {"job": {"id": f"test-db-{upload.file_id}"}}
    result = await process_sqlite(
        ctx,
        file_id=upload.file_id,
        user=request.user,
        business_context=request.business_context,
        sample_limit=2,
    )
    print(f"[db] completed -> schema={result['schema_path']}")


async def test_knowledge() -> None:
    ctx = {"job": {"id": "test-knowledge"}}
    result = await process_knowledge(
        ctx,
        user=TEST_USER,
        collection_name="test_knowledge",
        query="What is the operating model of the demo store?",
        answer="The demo store aggregates online and in-person sales with a unified catalog and shared fulfillment.",
        file_id=None,
        force=True,
    )
    print(f"[knowledge] completed -> {result}")


async def test_queue_roundtrip() -> None:
    """
    Optional queue-based roundtrip to verify SAQ wiring with staged uploads.
    """
    worker = await start_default_worker(concurrency=2)
    queue = get_queue()

    pdf_path = TEST_FILES_DIR / "ada.pdf"
    csv_path = TEST_FILES_DIR / "testing.csv"
    sqlite_path = TEST_FILES_DIR / "department_store.sqlite"
    schema_path = TEST_FILES_DIR / "schema.sql"

    pdf_upload = stage_upload(
        UploadRequest(
            user=TEST_USER,
            business_context=BUSINESS_CONTEXT,
            upload_type=UploadType.PDF,
            source_path=pdf_path,
        )
    )
    csv_upload = stage_upload(
        UploadRequest(
            user=TEST_USER,
            business_context=BUSINESS_CONTEXT,
            upload_type=UploadType.CSV,
            source_path=csv_path,
        )
    )
    db_upload = stage_upload(
        UploadRequest(
            user=TEST_USER,
            business_context=BUSINESS_CONTEXT,
            upload_type=UploadType.SQLITE,
            source_path=sqlite_path,
            schema_sql_path=schema_path,
        )
    )

    pdf_job = await enqueue_pdf_job(pdf_upload.file_id, TEST_USER, BUSINESS_CONTEXT, collection_name="test_docs")
    csv_job = await enqueue_csv_job(csv_upload.file_id, TEST_USER, BUSINESS_CONTEXT)
    db_job = await enqueue_sqlite_job(db_upload.file_id, TEST_USER, BUSINESS_CONTEXT)
    knowledge_job = await enqueue_knowledge_job(
        user=TEST_USER,
        collection_name="test_knowledge",
        query="How are docs processed?",
        answer="Docs flow through pre, post, and ingest steps with idempotent artifacts.",
    )

    print(f"[queue] enqueued jobs -> pdf={pdf_job}, csv={csv_job}, db={db_job}, knowledge={knowledge_job}")

    for _ in range(30):
        queued = await queue.count("queued")
        active = await queue.count("active")
        print(f"[queue] queued={queued} active={active}")
        if queued == 0 and active == 0:
            break
        await asyncio.sleep(0.5)

    await stop_worker(worker)
    print("[queue] all jobs processed (worker stopped)")


async def main() -> None:
    await test_pdf()
    await test_csv()
    await test_db()
    await test_knowledge()
    # Uncomment to verify queue end-to-end
    # await test_queue_roundtrip()


if __name__ == "__main__":
    asyncio.run(main())
