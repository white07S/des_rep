from __future__ import annotations

import asyncio
import hashlib
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[4]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from backend.module.logging import LoggingInterceptor  # noqa: E402
from backend.module.pipelines.docs.document_pipeline import run_document_pipeline  # noqa: E402

logger = LoggingInterceptor("document_pipeline_test")

# Hardcoded inputs (adjust as needed)
INPUT_PDF_PATH = Path("/Users/preetam/Develop/rag_business/backend/test_files/ada.pdf")
OUTPUT_PATH = Path("/Users/preetam/Develop/rag_business/backend/test_files/docs_output")
COLLECTION_NAME = "earning_report"
QDRANT_FILE_PATH = Path("/Users/preetam/Develop/rag_business/backend/test_files/qdrant")
CHUNK_MAPPING_PATH = Path("/Users/preetam/Develop/rag_business/backend/test_files/docs_chunks.sqlite")


def _hash_filename(name: str) -> str:
    """Hash the filename to build the output bucket path."""
    digest = hashlib.sha256(name.encode("utf-8")).hexdigest()
    logger.info("Computed name hash", name=name, hash=digest)
    return digest


def _write_bytes(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(payload)


def _write_text(target: Path, payload: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(payload, encoding="utf-8")


async def run() -> None:
    if not INPUT_PDF_PATH.exists():
        raise FileNotFoundError(f"Input PDF not found: {INPUT_PDF_PATH}")

    name_hash = _hash_filename(INPUT_PDF_PATH.name)
    bucket_dir = OUTPUT_PATH / name_hash
    bucket_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running document pipeline", input_pdf=str(INPUT_PDF_PATH), output_dir=str(bucket_dir))
    result = await run_document_pipeline(
        INPUT_PDF_PATH,
        collection_name=COLLECTION_NAME,
        qdrant_path=QDRANT_FILE_PATH,
        chunk_mapping_path=CHUNK_MAPPING_PATH,
    )
    stem = INPUT_PDF_PATH.stem

    pre_json_path = bucket_dir / f"pre_{stem}.json"
    pre_md_path = bucket_dir / f"pre_{stem}.md"
    post_md_path = bucket_dir / f"post_{stem}.md"

    _write_text(pre_md_path, result.pre.markdown)
    _write_text(post_md_path, result.post.processed_markdown)
    _write_text(pre_json_path, json.dumps(result.pre.analysis, ensure_ascii=False, indent=2))

    for image in result.pre.images:
        target = bucket_dir / image.suggested_path
        _write_bytes(target, image.content_bytes)

    print("\nDocument pipeline completed:")
    print(f"  Input PDF:      {INPUT_PDF_PATH}")
    print(f"  Output bucket:  {bucket_dir}")
    print(f"  Pre JSON:       {pre_json_path}")
    print(f"  Pre markdown:   {pre_md_path}")
    print(f"  Post markdown:  {post_md_path}")
    print(f"  Images saved:   {len(result.pre.images)}")
    print(f"  Qdrant path:    {QDRANT_FILE_PATH}")
    print(f"  Chunk mapping:  {CHUNK_MAPPING_PATH}")
    print(f"  Total chunks:   {result.ingest.total_chunks} (new: {result.ingest.inserted_chunks}, reused: {result.ingest.reused_chunks})")


if __name__ == "__main__":
    asyncio.run(run())
